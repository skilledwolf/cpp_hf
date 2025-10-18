// prof.hpp - lightweight aggregate timers, enabled via HF_ENABLE_PROFILING
#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <mutex>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace hf {

#if defined(HF_ENABLE_PROFILING)

struct ProfStats {
    std::uint64_t count{0};
    long double total_ns{0};
};

class ProfRegistry {
public:
    static ProfRegistry& inst() {
        static ProfRegistry r; return r;
    }
    void record(std::string_view name, long double ns) {
        std::lock_guard<std::mutex> lock(mu_);
        auto& s = stats_[std::string(name)];
        s.count += 1;
        s.total_ns += ns;
    }
    void reset() {
        std::lock_guard<std::mutex> lock(mu_);
        stats_.clear();
    }
    void dump(FILE* fp = stderr) {
        std::vector<std::pair<std::string, ProfStats>> v;
        {
            std::lock_guard<std::mutex> lock(mu_);
            v.reserve(stats_.size());
            for (auto& kv : stats_) v.emplace_back(kv.first, kv.second);
        }
        std::sort(v.begin(), v.end(), [](auto& a, auto& b){ return a.second.total_ns > b.second.total_ns; });
        std::fprintf(fp, "\n==== cpp_hf profile (aggregated) ====\n");
        std::fprintf(fp, "%-32s %10s %14s %12s\n", "section", "count", "total_ms", "avg_ms");
        for (auto& [name, s] : v) {
            const double tot_ms = static_cast<double>(s.total_ns) / 1.0e6;
            const double avg_ms = (s.count ? tot_ms / static_cast<double>(s.count) : 0.0);
            std::fprintf(fp, "%-32s %10llu %14.3f %12.3f\n",
                         name.c_str(), static_cast<unsigned long long>(s.count), tot_ms, avg_ms);
        }
        std::fflush(fp);
    }
    bool should_auto_dump() const {
        const char* env = std::getenv("HF_PROFILE");
        if (!env) return false; // enable only if explicitly requested
        // Any non-empty value except "0" enables
        return std::string(env) != "0";
    }
private:
    std::unordered_map<std::string, ProfStats> stats_;
    std::mutex mu_;
};

class ScopedTimer {
public:
    explicit ScopedTimer(std::string_view name) : name_(name), t0_(Clock::now()) {}
    ~ScopedTimer() {
        const auto t1 = Clock::now();
        const long double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0_).count();
        ProfRegistry::inst().record(name_, ns);
    }
private:
    using Clock = std::chrono::steady_clock;
    std::string name_;
    Clock::time_point t0_;
};

inline void prof_reset() { ProfRegistry::inst().reset(); }
inline void prof_dump()  { ProfRegistry::inst().dump(); }
inline bool prof_auto_dump_enabled() { return ProfRegistry::inst().should_auto_dump(); }

#  define HF_PROFILE_SCOPE(name_literal) ::hf::ScopedTimer HF_PP_JOIN(_hf_scope_timer_, __LINE__){name_literal}
#  define HF_PP_JOIN2(a,b) a##b
#  define HF_PP_JOIN(a,b) HF_PP_JOIN2(a,b)

#else

inline void prof_reset() {}
inline void prof_dump() {}
inline bool prof_auto_dump_enabled() { return false; }

#  define HF_PROFILE_SCOPE(name_literal) do{}while(0)

#endif

} // namespace hf

