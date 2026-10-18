#include "graphics/host_gpu/renderer/drawCensus.h"

#include "common/logging/log.h"
#include "graphics/shader/recompiler/ir/ResourceSnapshot.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/shader.h"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cstdlib>
#include <unordered_map>
#include <vector>
#include <xxhash.h>

namespace Libs::Graphics::DrawCensus {

namespace {

enum Class : uint8_t { Skipped = 0, Repeat = 1, SameFrame = 2, New = 3, ClassCount = 4 };

constexpr const char* ClassNames[ClassCount] = {"skipped", "repeat", "same-frame", "new"};

// A key is a record candidate while it was seen within this many frames.
constexpr uint64_t KeepFrames = 60;

struct Seen {
	uint64_t first_frame = 0;
	uint64_t last_frame  = 0;
};

// What a shape (programs, targets, topology: the key less the resources) last bound, to tell
// which resource kind changed when a known shape comes with a new key.
struct ShapeParts {
	uint64_t buffers  = 0;
	uint64_t images   = 0;
	uint64_t samplers = 0;
	uint64_t last_frame = 0;
};

struct Totals {
	uint64_t count[ClassCount] {};
	double   seconds[ClassCount] {};
	uint64_t exact = 0;
	// New keys of a shape seen before, by what differed from its last use (bit 0 buffers, 1
	// images, 2 samplers); 0: a known shape whose parts are all as before (another combination
	// of them).
	uint64_t new_known_shape[8] {};
	uint64_t new_shape = 0;
};

struct ShaderMisses {
	uint64_t count = 0;
	uint32_t changed_bits = 0; // union of the parts that differed
};

struct State {
	std::unordered_map<uint64_t, Seen>       keys[2];
	std::unordered_map<uint64_t, ShapeParts> shapes[2];
	Totals                                   totals[2];
	std::unordered_map<uint64_t, ShaderMisses> misses[2]; // by program pair hash
	std::chrono::steady_clock::time_point    next_log {};
	uint64_t                                 pruned_frame = 0;
};

State& GetState() {
	static State state;
	return state;
}

thread_local Scope* t_scope = nullptr;

uint64_t HashDescriptors(const std::vector<ShaderRecompiler::IR::DescriptorValue>& values,
                         uint64_t seed) {
	uint64_t hash = XXH3_64bits_withSeed(nullptr, 0, seed ^ values.size());
	for (const auto& value: values) {
		hash = XXH3_64bits_withSeed(value.dwords.data(), value.dword_count * sizeof(uint32_t), hash);
	}
	return hash;
}

uint64_t HashWords(const std::vector<uint32_t>& words, uint64_t seed) {
	return XXH3_64bits_withSeed(words.data(), words.size() * sizeof(uint32_t), seed ^ words.size());
}

struct StageParts {
	uint64_t program   = 0;
	uint64_t buffers   = 0;
	uint64_t images    = 0;
	uint64_t samplers  = 0;
	uint64_t constants = 0;
};

StageParts PartsOf(const ShaderStageRuntime& stage) {
	StageParts parts;
	const auto& resources = *stage.resources;
	parts.program  = XXH3_64bits(&stage.program, sizeof(stage.program));
	parts.buffers  = HashDescriptors(resources.buffers, 1);
	parts.images   = HashDescriptors(resources.images, 2);
	parts.samplers = HashDescriptors(resources.samplers, 3);
	// Bindless heaps decide the images a draw samples as much as its own descriptors do.
	parts.images = XXH3_64bits_withSeed(resources.bindless_heaps.data(),
	                                    resources.bindless_heaps.size() *
	                                        sizeof(ShaderRecompiler::IR::BindlessHeapUse),
	                                    parts.images);
	parts.constants = HashWords(resources.user_data, HashWords(resources.flattened_srt, 4));
	return parts;
}

uint64_t Mix(uint64_t hash, uint64_t value) {
	return XXH3_64bits_withSeed(&value, sizeof(value), hash);
}

void Log(State& state) {
	const auto now = std::chrono::steady_clock::now();
	if (now < state.next_log) {
		return;
	}
	const bool first = state.next_log == std::chrono::steady_clock::time_point {};
	state.next_log   = now + std::chrono::seconds(10);
	if (first) {
		return;
	}
	for (uint32_t kind = 0; kind < 2; kind++) {
		auto&          t     = state.totals[kind];
		uint64_t       total = 0;
		double         seconds = 0.0;
		for (uint32_t c = 0; c < ClassCount; c++) {
			total += t.count[c];
			seconds += t.seconds[c];
		}
		if (total == 0) {
			continue;
		}
		const auto pct = [&](uint64_t n) { return 100.0 * static_cast<double>(n) / static_cast<double>(total); };
		const auto avg = [&](uint32_t c) {
			return t.count[c] == 0 ? 0.0 : t.seconds[c] * 1e6 / static_cast<double>(t.count[c]);
		};
		const auto prepared = total - t.count[Skipped];
		LOGF("Draw census (10 s) %s: %" PRIu64 " (%.0f ms on the GPU thread), repeat %.1f%% (%.1f us "
		     "avg, %.0f ms), same-frame %.1f%% (%.1f us), new %.1f%% (%.1f us), skipped %.1f%% (%.1f "
		     "us); exact %.1f%% of repeats; %zu keys\n",
		     kind == 0 ? "draws" : "dispatches", total, seconds * 1000.0, pct(t.count[Repeat]),
		     avg(Repeat), t.seconds[Repeat] * 1000.0, pct(t.count[SameFrame]), avg(SameFrame),
		     pct(t.count[New]), avg(New), pct(t.count[Skipped]), avg(Skipped),
		     t.count[Repeat] == 0 ? 0.0
		                          : 100.0 * static_cast<double>(t.exact) / static_cast<double>(t.count[Repeat]),
		     state.keys[kind].size());
		if (prepared != 0) {
			const auto& n = t.new_known_shape;
			LOGF("Draw census (10 s) %s, new keys: new shape %" PRIu64 "; known shape with "
			     "buffers changed %" PRIu64 ", images %" PRIu64 ", samplers %" PRIu64
			     ", buffers+images %" PRIu64 ", other combinations %" PRIu64 ", same parts %" PRIu64 "\n",
			     kind == 0 ? "draws" : "dispatches", t.new_shape, n[1], n[2], n[4], n[3],
			     n[5] + n[6] + n[7], n[0]);
		}
		std::vector<std::pair<uint64_t, ShaderMisses>> top(state.misses[kind].begin(),
		                                                   state.misses[kind].end());
		std::sort(top.begin(), top.end(),
		          [](const auto& a, const auto& b) { return a.second.count > b.second.count; });
		for (size_t i = 0; i < std::min<size_t>(top.size(), 6); i++) {
			const auto bits = top[i].second.changed_bits;
			LOGF("\t new keys of a known shape: shader %016" PRIx64 " x%" PRIu64 " (%s%s%s)\n",
			     top[i].first, top[i].second.count, (bits & 1u) != 0 ? "buffers " : "",
			     (bits & 2u) != 0 ? "images " : "", (bits & 4u) != 0 ? "samplers" : "");
		}
		t = {};
		state.misses[kind].clear();
	}
}

void Prune(State& state, uint64_t frame) {
	if (frame < state.pruned_frame + 600) {
		return;
	}
	state.pruned_frame = frame;
	for (uint32_t kind = 0; kind < 2; kind++) {
		std::erase_if(state.keys[kind],
		              [&](const auto& entry) { return entry.second.last_frame + KeepFrames < frame; });
		std::erase_if(state.shapes[kind],
		              [&](const auto& entry) { return entry.second.last_frame + KeepFrames < frame; });
	}
}

void Classify(uint32_t kind, uint64_t key, uint64_t exact_key, uint64_t shape, uint64_t shape_name,
              const StageParts& parts, uint64_t frame) {
	auto& state = GetState();
	Prune(state, frame);
	auto& seen   = state.keys[kind][key];
	auto  cls    = New;
	if (seen.last_frame != 0 && seen.last_frame + KeepFrames >= frame) {
		cls = seen.first_frame < frame ? Repeat : SameFrame;
	}
	if (cls == New) {
		seen.first_frame = frame;
	}
	seen.last_frame = frame;
	// The exact key: the same key and constants in an earlier frame.
	auto& exact_seen = state.keys[kind][exact_key ^ 0x9e3779b97f4a7c15ull];
	const bool exact = cls == Repeat && exact_seen.last_frame != 0 && exact_seen.first_frame < frame &&
	                   exact_seen.last_frame + KeepFrames >= frame;
	if (exact_seen.last_frame == 0 || exact_seen.last_frame + KeepFrames < frame) {
		exact_seen.first_frame = frame;
	}
	exact_seen.last_frame = frame;

	auto& totals = state.totals[kind];
	totals.exact += exact ? 1u : 0u;
	auto& known = state.shapes[kind][shape];
	if (cls == New) {
		if (known.last_frame == 0) {
			totals.new_shape++;
		} else {
			const uint32_t bits = (known.buffers != parts.buffers ? 1u : 0u) |
			                      (known.images != parts.images ? 2u : 0u) |
			                      (known.samplers != parts.samplers ? 4u : 0u);
			totals.new_known_shape[bits]++;
			auto& misses = state.misses[kind][shape_name];
			misses.count++;
			misses.changed_bits |= bits;
		}
	}
	known.buffers    = parts.buffers;
	known.images     = parts.images;
	known.samplers   = parts.samplers;
	known.last_frame = frame;
	if (t_scope != nullptr) {
		t_scope->Set(cls, exact);
	}
}

} // namespace

bool Enabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_DRAW_CENSUS");
		const bool  on    = value != nullptr && value[0] == '1';
		if (on) {
			LOGF("Draw census on (KYTY_DRAW_CENSUS=1): draws and dispatches keyed and timed, "
			     "a report every 10 s\n");
		}
		return on;
	}();
	return enabled;
}

Scope::Scope(Kind kind): m_kind(kind) {
	if (!Enabled()) {
		return;
	}
	m_active = true;
	m_outer  = t_scope;
	t_scope  = this;
	m_start  = std::chrono::steady_clock::now();
}

Scope::~Scope() {
	if (!m_active) {
		return;
	}
	const double seconds =
	    std::chrono::duration<double>(std::chrono::steady_clock::now() - m_start).count();
	t_scope     = m_outer;
	auto& state = GetState();
	auto& t     = state.totals[m_kind == Kind::Draw ? 0 : 1];
	t.count[m_class]++;
	t.seconds[m_class] += seconds;
	if (m_outer == nullptr) {
		Log(state);
	}
}

void ClassifyDraw(std::span<const ShaderStageRuntime* const> stages,
                  std::span<const uint64_t> targets, uint32_t topology, uint64_t frame) {
	if (!Enabled() || t_scope == nullptr) {
		return;
	}
	uint64_t   key = 0x6472617721ull, exact = key, shape = key, name = 0;
	StageParts all;
	for (const auto* stage: stages) {
		if (stage == nullptr || !*stage) {
			continue;
		}
		const auto parts = PartsOf(*stage);
		shape            = Mix(shape, parts.program);
		key   = Mix(Mix(Mix(Mix(key, parts.program), parts.buffers), parts.images), parts.samplers);
		exact = Mix(exact, parts.constants);
		all.buffers  = Mix(all.buffers, parts.buffers);
		all.images   = Mix(all.images, parts.images);
		all.samplers = Mix(all.samplers, parts.samplers);
		name         = stage->program->shader_hash; // the last stage: the pixel shader if any
	}
	for (const auto target: targets) {
		shape = Mix(shape, target);
	}
	shape = Mix(shape, topology);
	key   = Mix(key, shape);
	exact = Mix(exact, key);
	Classify(0, key, exact, shape, name, all, frame);
}

void ClassifyDispatch(const ShaderStageRuntime& stage, uint64_t frame) {
	if (!Enabled() || t_scope == nullptr || !stage) {
		return;
	}
	const auto parts = PartsOf(stage);
	const auto shape = parts.program;
	const auto key   = Mix(Mix(Mix(parts.program, parts.buffers), parts.images), parts.samplers);
	const auto exact = Mix(key, parts.constants);
	Classify(1, key, exact, shape, stage.program->shader_hash, parts, frame);
}

} // namespace Libs::Graphics::DrawCensus
