#include "graphics/host_gpu/frameBudget.h"

#include "common/logging/log.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <utility>
#include <vector>
#include <fmt/format.h>

#if defined(_M_X64) || defined(__x86_64__)
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#define KYTY_FRAME_BUDGET_TSC 1
#endif

namespace Libs::Graphics::FrameBudget {

namespace {

constexpr size_t PhaseCount = static_cast<size_t>(Phase::Count);

constexpr std::array<const char*, PhaseCount> PhaseNames = {
    "other",  "idle",     "blocked",  "requests",  "pm4",     "draw",    "dispatch",
    "shaders", "buffers", "textures", "readback",  "wait-gpu", "wait-rec", "collect",
    "flip",   "present",  "record",   "submit",    "guest-wait"};

// Ticks: the time stamp counter where there is one (a few ns a read, against the clock's ~20),
// converted with the clock measured over each report.
inline uint64_t Ticks() {
#ifdef KYTY_FRAME_BUDGET_TSC
	return __rdtsc();
#else
	return static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
#endif
}

struct ThreadState {
	std::atomic<const char*>                       name {nullptr};
	std::atomic<uint8_t>                           current {0};
	std::atomic<uint64_t>                          since {0};
	std::array<std::atomic<uint64_t>, PhaseCount>  ticks {};
	std::array<uint64_t, PhaseCount>               reported {}; // reporter only
};

struct Registry {
	std::mutex              mutex;
	std::deque<ThreadState> threads; // stable addresses
};

Registry& GetRegistry() {
	static Registry registry;
	return registry;
}

constexpr size_t MaxThreads = 512;
ThreadState      g_overflow; // threads past MaxThreads share it (not reported)

ThreadState& Local() {
	static thread_local ThreadState* state = nullptr;
	if (state == nullptr) {
		auto&           registry = GetRegistry();
		std::lock_guard lock(registry.mutex);
		if (registry.threads.size() < MaxThreads) {
			state = &registry.threads.emplace_back();
			state->since.store(Ticks(), std::memory_order_relaxed);
		} else {
			state = &g_overflow;
		}
	}
	return *state;
}

inline void Switch(ThreadState& thread, Phase from, Phase to) {
	const uint64_t now   = Ticks();
	const uint64_t since = thread.since.load(std::memory_order_relaxed);
	auto&          slot  = thread.ticks[static_cast<size_t>(from)];
	// One writer per thread: a load and a store, no locked instruction.
	slot.store(slot.load(std::memory_order_relaxed) + (now - since), std::memory_order_relaxed);
	thread.since.store(now, std::memory_order_relaxed);
	thread.current.store(static_cast<uint8_t>(to), std::memory_order_relaxed);
}

struct ReportState {
	std::mutex                            mutex;
	uint64_t                              frames     = 0;
	uint64_t                              last_ticks = 0;
	std::chrono::steady_clock::time_point last_time {};
	bool                                  started = false;
};

ReportState g_report;

using Groups = std::vector<std::pair<std::string, std::array<uint64_t, PhaseCount>>>;

// Per name: the phases' ticks since the last call.
Groups CollectWindow() {
	const uint64_t now = Ticks();
	Groups         groups;
	{
		auto&           registry = GetRegistry();
		std::lock_guard lock(registry.mutex);
		for (auto& thread: registry.threads) {
			std::array<uint64_t, PhaseCount> totals {};
			for (size_t i = 0; i < PhaseCount; i++) {
				totals[i] = thread.ticks[i].load(std::memory_order_relaxed);
			}
			// The phase in progress is credited up to now (and not again next time: the window
			// is measured against what was reported, credit included).
			const auto     current = thread.current.load(std::memory_order_relaxed);
			const uint64_t since   = thread.since.load(std::memory_order_relaxed);
			if (current < PhaseCount && now > since) {
				totals[current] += now - since;
			}
			std::array<uint64_t, PhaseCount> window {};
			for (size_t i = 0; i < PhaseCount; i++) {
				window[i]          = totals[i] > thread.reported[i] ? totals[i] - thread.reported[i] : 0;
				thread.reported[i] = std::max(totals[i], thread.reported[i]);
			}
			const char* name  = thread.name.load(std::memory_order_relaxed);
			const bool  named = name != nullptr;
			if (!named) {
				window[static_cast<size_t>(Phase::Other)] = 0;
				window[static_cast<size_t>(Phase::Idle)]  = 0;
			}
			const std::string key = named ? name : "others";
			auto it = std::find_if(groups.begin(), groups.end(),
			                       [&](const auto& group) { return group.first == key; });
			if (it == groups.end()) {
				groups.emplace_back(key, std::array<uint64_t, PhaseCount> {});
				it = groups.end() - 1;
			}
			for (size_t i = 0; i < PhaseCount; i++) {
				it->second[i] += window[i];
			}
		}
	}
	return groups;
}

void Report(uint64_t frames, double seconds, uint64_t elapsed_ticks) {
	auto groups = CollectWindow();
	if (frames == 0 || seconds <= 0.0 || elapsed_ticks == 0) {
		return;
	}
	const double ms_per_tick = seconds * 1000.0 / static_cast<double>(elapsed_ticks);
	const double per_frame   = 1.0 / static_cast<double>(frames);
	std::sort(groups.begin(), groups.end(), [](const auto& a, const auto& b) {
		const auto rank = [](const std::string& name) {
			return name == "gpu" ? 0 : name == "record" ? 1 : name == "submit" ? 2 : name == "others" ? 9 : 5;
		};
		return rank(a.first) != rank(b.first) ? rank(a.first) < rank(b.first) : a.first < b.first;
	});

	const double frame_ms = seconds * 1000.0 * per_frame;
	std::string  text = fmt::format("Frame budget ({:.0f} s): {} frames, {:.1f} fps, {:.1f} ms/frame; "
	                                "ms per frame, by thread:\n",
	                                seconds, frames, static_cast<double>(frames) / seconds, frame_ms);
	for (const auto& [name, ticks]: groups) {
		std::vector<std::pair<double, size_t>> phases;
		double                                 busy = 0.0;
		for (size_t i = 0; i < PhaseCount; i++) {
			const double ms = static_cast<double>(ticks[i]) * ms_per_tick * per_frame;
			if (i != static_cast<size_t>(Phase::Idle)) {
				busy += ms;
			}
			if (ms >= 0.05) {
				phases.emplace_back(ms, i);
			}
		}
		if (phases.empty()) {
			continue;
		}
		std::sort(phases.begin(), phases.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
		text += fmt::format("  {}:", name);
		for (const auto& [ms, i]: phases) {
			text += fmt::format(" {} {:.1f}", PhaseNames[i], ms);
		}
		if (name != "others") {
			text += fmt::format(" (busy {:.1f})", busy);
		}
		text += "\n";
	}
	LOGF("%s", text.c_str());
}

} // namespace

bool Enabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_FRAME_BUDGET");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

namespace Detail {

Phase Enter(Phase phase) {
	auto&       thread   = Local();
	const auto  previous = static_cast<Phase>(thread.current.load(std::memory_order_relaxed));
	if (previous != phase) {
		Switch(thread, previous, phase);
	}
	return previous;
}

void Leave(Phase previous) {
	auto&      thread  = Local();
	const auto current = static_cast<Phase>(thread.current.load(std::memory_order_relaxed));
	if (current != previous) {
		Switch(thread, current, previous);
	}
}

} // namespace Detail

void NameThread(const char* name) {
	if (Enabled()) {
		Local().name.store(name, std::memory_order_relaxed);
	}
}

void Present() {
	if (!Enabled()) {
		return;
	}
	constexpr auto  Interval = std::chrono::seconds(5);
	const auto      now      = std::chrono::steady_clock::now();
	std::lock_guard lock(g_report.mutex);
	if (!g_report.started) {
		g_report.started    = true;
		g_report.last_time  = now;
		g_report.last_ticks = Ticks();
		g_report.frames     = 0;
		(void)CollectWindow(); // the first window starts here
		return;
	}
	g_report.frames++;
	if (now - g_report.last_time < Interval) {
		return;
	}
	const uint64_t ticks   = Ticks();
	const double   seconds = std::chrono::duration<double>(now - g_report.last_time).count();
	Report(g_report.frames, seconds, ticks - g_report.last_ticks);
	g_report.frames     = 0;
	g_report.last_time  = now;
	g_report.last_ticks = ticks;
}

} // namespace Libs::Graphics::FrameBudget
