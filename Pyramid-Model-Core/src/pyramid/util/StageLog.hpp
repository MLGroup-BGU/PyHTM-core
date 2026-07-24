/* ---------------------------------------------------------------------------
 * StageLog.hpp -- colored stage banners for the pyramid engine's stdout.
 *
 * Each engine phase announces itself with a bold, distinctly-colored banner
 * so the log stream reads as clearly separated stages:
 *
 *   ENCODERS -> ARCHITECTURE -> WORKERS -> INIT (SP+TM) -> RUN -> DONE
 *
 * Pure C++ (printf), safe to call GIL-free.  ANSI escape sequences; on
 * Windows, virtual-terminal processing is enabled once per process (Windows
 * 10+); if that fails, or when stdout is not a terminal, banners fall back
 * to plain uncolored text.
 * ------------------------------------------------------------------------- */
#pragma once

#include <cstdio>
#include <cstdlib>
#include <string>
#include <functional>

#if defined(_WIN32)
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
  #include <io.h>
  #define HTM_ISATTY(fd) _isatty(fd)
  #define HTM_FILENO(f)  _fileno(f)
#else
  #include <unistd.h>
  #define HTM_ISATTY(fd) isatty(fd)
  #define HTM_FILENO(f)  fileno(f)
#endif

namespace htm_pyramid {

/* Format seconds as H:MM:SS (or M:SS under an hour) -- so progress reads as
 * elapsed/remaining wall-clock, not a raw second count. */
/* Process-global interrupt hook. The Python binding installs a function that
 * checks for pending signals (Ctrl+C) under a brief GIL acquire and throws if
 * one is pending; the run loops call it on their progress cadence so a
 * GIL-released C++ run remains interruptible. Default: no-op. */
// A plain C++ exception used to signal Ctrl+C out of GIL-released C++ code.
// We deliberately do NOT throw pybind11::error_already_set here: that object
// carries Python references and must be created/destroyed with the GIL held,
// which is fragile when the exception unwinds through GIL-released worker code
// and pool destructors. Instead we throw this trivial type and let the binding
// translate it into KeyboardInterrupt at the top, with the GIL held.
struct PyramidInterrupted {};

/* -------------------------------------------------------------------------
 * PROGRESS_REFRESH_SECONDS -- how often the "[Run] ..." progress line is
 * redrawn, in seconds. USER-TUNABLE: edit this one value and rebuild.
 *   0.05  = 20 refreshes per second (default -- smooth, negligible cost)
 *   0.02  = 50 refreshes per second
 *   0.10  = 10 refreshes per second
 * Applies to both the barrier and the pipeline run paths.
 * ------------------------------------------------------------------------- */
inline constexpr double PROGRESS_REFRESH_SECONDS = 0.05;

/* Sliding-window rate meter for the progress line. The old display divided
 * total-done by total-elapsed -- a since-start AVERAGE that keeps sinking
 * long after the instantaneous speed has stabilised (and skews the ETA the
 * same way). This keeps the last ~6 seconds of (time, done) samples and
 * reports the rate across that window, so the number on screen is the
 * CURRENT speed; the run's true average still appears in the final summary
 * line. Falls back to the cumulative rate until the window has >= 0.5 s. */
struct RateWindow {
    static constexpr int N = 128;                 // ~6.4 s at 20 samples/s
    double t[N]; long long d[N]; int head = 0, count = 0;
    void add(double now, long long done) {
        t[head] = now; d[head] = done;
        head = (head + 1) % N;
        if (count < N) ++count;
    }
    double rate(double now, long long done, double fallback) const {
        if (count < 2) return fallback;
        const int oldest = (head - count + N) % N;
        const double dt = now - t[oldest];
        if (dt < 0.5) return fallback;
        return static_cast<double>(done - d[oldest]) / dt;
    }
};

inline std::function<void()> &interrupt_hook() {
    static std::function<void()> hook;
    return hook;
}
inline void check_interrupt() {
    auto &h = interrupt_hook();
    if (h) h();
}

inline std::string fmt_hms(double seconds) {
    if (seconds < 0 || !(seconds == seconds)) seconds = 0;  // guard NaN/neg
    long long s = static_cast<long long>(seconds + 0.5);
    long long h = s / 3600; s %= 3600;
    long long m = s / 60;   s %= 60;
    char buf[48];
    // Explicit unit labels; omit larger units when zero (no "00h" / leading
    // "00m") so short runs read "12s" or "3m 04s", long runs "1h 05m 23s".
    if (h > 0)
        std::snprintf(buf, sizeof(buf), "%lldh %02lldm %02llds", h, m, s);
    else if (m > 0)
        std::snprintf(buf, sizeof(buf), "%lldm %02llds", m, s);
    else
        std::snprintf(buf, sizeof(buf), "%llds", s);
    return std::string(buf);
}

/** Stage identifiers -- one color per stage. */
enum class Stage {
    Encoders,       // cyan    -- feature encoders (+ sample pass)
    Architecture,   // magenta -- node construction, routing, slots
    Workers,        // yellow  -- core discovery + thread<->model assignment
    Init,           // green   -- SP + TM initialization (parallel)
    Run,            // blue    -- record streaming
    Done            // bright green -- completion summary
};

/* Is stdout a terminal?
 *
 * The progress line is redrawn in place with a carriage return, which only
 * makes sense on a terminal. Sent to a file -- an sbatch output log, a pipe,
 * a redirect -- those carriage returns pile up into one enormous logical line
 * that no reader displays sensibly. Callers use this to choose between
 * redrawing and appending. */
inline bool stdout_is_terminal() {
    static const bool tty = HTM_ISATTY(HTM_FILENO(stdout)) != 0;
    return tty;
}

/* Percent step between progress lines when stdout is NOT a terminal.
 *
 * A file wants a readable trail, not an animation: at the default of 1 a run
 * leaves about a hundred lines whatever its length. Override with
 * PYRAMID_PROGRESS_PCT, an integer from 1 to 100. Out-of-range or unparsable
 * values fall back to the default rather than failing the run. */
inline int progress_percent_step() {
    static const int step = [] {
        if (const char *env = std::getenv("PYRAMID_PROGRESS_PCT")) {
            const int v = std::atoi(env);
            if (v >= 1 && v <= 100) return v;
        }
        return 1;
    }();
    return step;
}

namespace detail {
inline bool ansi_ok() {
    static const bool ok = [] {
        if (!stdout_is_terminal()) return false;
#if defined(_WIN32)
        HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
        DWORD mode = 0;
        if (h == INVALID_HANDLE_VALUE || !GetConsoleMode(h, &mode))
            return false;
        SetConsoleOutputCP(CP_UTF8);   /* ASCII-safe output regardless of
                                          the console's legacy codepage */
        return SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING)
               != 0;
#else
        return true;
#endif
    }();
    return ok;
}

inline const char *color_of(Stage s) {
    switch (s) {
        case Stage::Encoders:     return "\033[1;36m";   // bold cyan
        case Stage::Architecture: return "\033[1;35m";   // bold magenta
        case Stage::Workers:      return "\033[1;33m";   // bold yellow
        case Stage::Init:         return "\033[1;32m";   // bold green
        case Stage::Run:          return "\033[1;34m";   // bold blue
        case Stage::Done:         return "\033[1;92m";   // bold bright green
    }
    return "";
}
}  // namespace detail

/* Renders the progress bar into `buf` as a SOLID FILL that grows:
 *   VT/UTF-8 terminals: [██████████▍             ]
 *       tqdm-style: the tip advances in 1/8-cell steps (U+2588 full block
 *       plus the partial blocks U+2589..U+258F), i.e. width*8 steps overall
 *       -- at width 24 the bar visibly moves every ~0.5% and full cells
 *       merge into one solid, continuously growing rectangle.
 *   plain fallback:     [##########..............]
 * Block glyphs are 3 UTF-8 bytes, so `buf` must hold at least
 * 3*width + 3 bytes ('[', ']', NUL); callers use width 24 with an
 * 80-byte buffer. Off-terminal output (redirected runs, log files)
 * stays pure ASCII, matching the banner policy above. */
inline void progress_bar(double frac, char *buf, int width) {
    if (frac < 0) frac = 0;
    if (frac > 1) frac = 1;
    const bool utf8 = detail::ansi_ok();
    int i = 0;
    buf[i++] = '[';
    if (utf8) {
        const int eighths = static_cast<int>(frac * width * 8 + 0.5);
        const int full    = eighths / 8;   /* fully filled cells          */
        const int rem     = eighths % 8;   /* 1..7 -> partial tip cell    */
        for (int b = 0; b < width; ++b) {
            if (b < full) {                        /* U+2588 FULL BLOCK   */
                buf[i++] = '\xE2'; buf[i++] = '\x96'; buf[i++] = '\x88';
            } else if (b == full && rem > 0) {     /* U+2589..U+258F tip  */
                buf[i++] = '\xE2'; buf[i++] = '\x96';
                buf[i++] = static_cast<char>(0x88 + (8 - rem));
            } else {
                buf[i++] = ' ';
            }
        }
    } else {
        const int filled = static_cast<int>(frac * width + 0.5);
        for (int b = 0; b < width; ++b)
            buf[i++] = (b < filled) ? '#' : '.';
    }
    buf[i++] = ']';
    buf[i] = '\0';
}

/** Print a bold colored banner announcing that subsequent logs belong to
 *  stage `title`. */
inline void stage_banner(Stage s, const char *title) {
    if (detail::ansi_ok())
        std::printf("\n%s====== STAGE: %s ======\033[0m\n",
                    detail::color_of(s), title);
    else
        std::printf("\n====== STAGE: %s ======\n", title);
    std::fflush(stdout);
}

/** Print an unmissable RED error line (plain fallback off-terminal). */
inline void error_line(const char *what) {
    if (detail::ansi_ok())
        std::printf("\n\033[1;31m[ERROR] %s\033[0m\n", what);
    else
        std::printf("\n[ERROR] %s\n", what);
    std::fflush(stdout);
}

}  // namespace htm_pyramid