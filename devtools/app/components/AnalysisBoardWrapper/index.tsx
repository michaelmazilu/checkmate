"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import Script from "next/script";
import { Figtree } from "next/font/google";

const figtree = Figtree({
  subsets: ["latin"],
  weight: ["600", "700"],
});

type TimeControl = 1 | 3 | 5 | 10;

export default function AnalysisBoardWrapper() {
  const [scriptsLoaded, setScriptsLoaded] = useState({
    jquery: false,
    analysis: false,
  });
  const boardContainerRef = useRef<HTMLDivElement>(null);
  const chessInstanceRef = useRef<any>(null);
  const [boardInitialized, setBoardInitialized] = useState(false);
  const [reloading, setReloading] = useState(false);
  const [engineError, setEngineError] = useState<string | null>(null);
  const [reloadStatus, setReloadStatus] = useState<string | null>(null);
  const [gameResult, setGameResult] = useState<{
    outcome: "win" | "lose";
    reason: string;
  } | null>(null);
  const lastRestartAtRef = useRef<number | null>(null);

  // Timer states
  const [timeControl, setTimeControl] = useState<TimeControl>(10);
  const [whiteTime, setWhiteTime] = useState(10 * 60 * 1000); // 10 minutes in ms
  const [blackTime, setBlackTime] = useState(10 * 60 * 1000);
  const [isRunning, setIsRunning] = useState(false);
  const [currentTurn, setCurrentTurn] = useState<"white" | "black">("white");
  const timerIntervalRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    if (typeof window === "undefined") return;

    const hasJQuery = Boolean((window as any).jQuery || (window as any).$);
    const hasInit = typeof (window as any).initAnalysisBoard === "function";

    if (!scriptsLoaded.jquery && hasJQuery) {
      setScriptsLoaded((prev) => ({ ...prev, jquery: true }));
    }

    if (!scriptsLoaded.analysis && hasInit) {
      setScriptsLoaded((prev) => ({ ...prev, analysis: true }));
    }
  }, [scriptsLoaded.jquery, scriptsLoaded.analysis]);

  useEffect(() => {
    if (typeof window === "undefined") return;

    const handler = (event: ErrorEvent) => {
      const message = event.message || String((event as any).error || "");
      const filename = event.filename || "";
      const isEngineFile = filename.includes(
        "/analysis-board/src/engines/stockfish"
      );
      const mentionsSharedArrayBuffer = message.includes("SharedArrayBuffer");
      const mentionsStockfish = message.toLowerCase().includes("stockfish");

      if (isEngineFile || mentionsSharedArrayBuffer || mentionsStockfish) {
        setEngineError(
          "Engine failed to load in this browser. Some Stockfish versions may not be supported."
        );
      }
    };

    window.addEventListener("error", handler);
    return () => {
      window.removeEventListener("error", handler);
    };
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") return;

    let cancelled = false;

    const fetchStatus = async () => {
      try {
        const res = await fetch("/api/reload/status");
        if (!res.ok) return;
        const data = await res.json();
        if (cancelled) return;
        if (typeof data.lastRestartAt === "number") {
          lastRestartAtRef.current = data.lastRestartAt;
        }
      } catch {}
    };

    fetchStatus();

    const interval = setInterval(async () => {
      try {
        const res = await fetch("/api/reload/status");
        if (!res.ok) return;
        const data = await res.json();
        if (cancelled) return;
        if (typeof data.lastRestartAt === "number") {
          const prev = lastRestartAtRef.current;
          const current = data.lastRestartAt as number;
          if (!prev || current > prev) {
            lastRestartAtRef.current = current;
            const reason = data.lastRestartReason as string | null | undefined;
            if (reason === "manual") {
              setReloadStatus("Server reloaded");
            } else if (reason === "file-change") {
              setReloadStatus("Server reloaded after file change");
            } else {
              setReloadStatus("Server reloaded");
            }
            setTimeout(() => {
              setReloadStatus(null);
            }, 3000);
          }
        }
      } catch {}
    }, 3000);

    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  const handleGameResult = useCallback(
    (result: { outcome?: string; reason?: string } | null) => {
      if (!result) return;
      setIsRunning(false); // Stop timer on game end
      setGameResult({
        outcome: result.outcome === "win" ? "win" : "lose",
        reason: result.reason || "checkmate",
      });
    },
    []
  );

  const handleRetry = useCallback(() => {
    setGameResult(null);
    if (typeof window !== "undefined") {
      window.location.reload();
    }
  }, []);

  // Format time as MM:SS
  const formatTime = useCallback((ms: number): string => {
    const totalSeconds = Math.max(0, Math.ceil(ms / 1000));
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    return `${minutes}:${seconds.toString().padStart(2, "0")}`;
  }, []);

  // Handle time control change
  const handleTimeControlChange = useCallback((minutes: TimeControl) => {
    setTimeControl(minutes);
    const timeInMs = minutes * 60 * 1000;
    setWhiteTime(timeInMs);
    setBlackTime(timeInMs);
    setIsRunning(false);
    setCurrentTurn("white");

    // Stop any running timer
    if (timerIntervalRef.current) {
      clearInterval(timerIntervalRef.current);
      timerIntervalRef.current = null;
    }
  }, []);

  // Timer countdown effect
  useEffect(() => {
    if (!isRunning) return;

    timerIntervalRef.current = setInterval(() => {
      if (currentTurn === "white") {
        setWhiteTime((prev) => {
          const newTime = prev - 100;
          if (newTime <= 0) {
            setIsRunning(false);
            setGameResult({ outcome: "lose", reason: "timeout" });
            return 0;
          }
          return newTime;
        });
      } else {
        setBlackTime((prev) => {
          const newTime = prev - 100;
          if (newTime <= 0) {
            setIsRunning(false);
            setGameResult({ outcome: "win", reason: "timeout" });
            return 0;
          }
          return newTime;
        });
      }
    }, 100);

    return () => {
      if (timerIntervalRef.current) {
        clearInterval(timerIntervalRef.current);
      }
    };
  }, [isRunning, currentTurn]);

  // Force clock display update to prevent external JS from overwriting
  useEffect(() => {
    if (!isRunning) return;

    const updateClockDisplays = () => {
      const blackClockTime = document.querySelector("#black-clock .clock-time");
      const whiteClockTime = document.querySelector("#white-clock .clock-time");

      if (
        blackClockTime &&
        blackClockTime.textContent !== formatTime(blackTime)
      ) {
        blackClockTime.textContent = formatTime(blackTime);
      }
      if (
        whiteClockTime &&
        whiteClockTime.textContent !== formatTime(whiteTime)
      ) {
        whiteClockTime.textContent = formatTime(whiteTime);
      }
    };

    // Update immediately
    updateClockDisplays();

    // Use MutationObserver to catch and immediately fix any external changes
    const observer = new MutationObserver(() => {
      updateClockDisplays();
    });

    // Observe both clock containers for any changes
    const blackClock = document.getElementById("black-clock");
    const whiteClock = document.getElementById("white-clock");

    if (blackClock) {
      observer.observe(blackClock, {
        childList: true,
        subtree: true,
        characterData: true,
      });
    }
    if (whiteClock) {
      observer.observe(whiteClock, {
        childList: true,
        subtree: true,
        characterData: true,
      });
    }

    return () => {
      observer.disconnect();
    };
  }, [isRunning, whiteTime, blackTime, formatTime]);

  const initializeAnalysisBoard = useCallback(async () => {
    try {
      const gameDataForAnalysis = {
        pgn: "",
        white: {
          name: "You",
          elo: "N/A",
        },
        black: {
          name: "Bot",
          elo: "N/A",
        },
        mode: "play-vs-bot",
        timeleft: timeControl * 60 * 1000, // Use selected time control
        playerColor: "white",
        onGameEnd: handleGameResult,
      };

      console.log("AnalysisBoardWrapper: initializing board", {
        playerColor: gameDataForAnalysis.playerColor,
      });

      (window as any).gameDataForAnalysis = gameDataForAnalysis;

      // wait on init function available
      if (typeof (window as any).initAnalysisBoard === "function") {
        if (chessInstanceRef.current?.destroy) {
          chessInstanceRef.current.destroy();
          chessInstanceRef.current = null;
        }
        const instance = await (window as any).initAnalysisBoard(
          gameDataForAnalysis
        );
        chessInstanceRef.current = instance;

        // Set up move listener to handle timer
        if (instance?.chess) {
          const originalPush = instance.chess.move.bind(instance.chess);
          instance.chess.move = function (...args: any[]) {
            const result = originalPush(...args);
            if (result) {
              // Start timer on first move
              setIsRunning(true);
              // Switch turns
              setCurrentTurn((prev) => (prev === "white" ? "black" : "white"));
            }
            return result;
          };
        }
      } else {
        console.error("initAnalysisBoard function not found on window");
      }
    } catch (error) {
      console.error("Failed to initialize analysis board:", error);
    }
  }, [handleGameResult, timeControl]);

  useEffect(() => {
    // init
    if (scriptsLoaded.jquery && scriptsLoaded.analysis && !boardInitialized) {
      setBoardInitialized(true);
      initializeAnalysisBoard();
    }
  }, [
    scriptsLoaded.jquery,
    scriptsLoaded.analysis,
    boardInitialized,
    initializeAnalysisBoard,
  ]);

  useEffect(() => {
    return () => {
      if (chessInstanceRef.current?.destroy) {
        chessInstanceRef.current.destroy();
        chessInstanceRef.current = null;
      }
    };
  }, []);

  const handleReloadClick = useCallback(async () => {
    if (reloading) return;
    setReloading(true);
    console.log("User pressed manual reload button");
    try {
      await fetch("/api/reload", {
        method: "POST",
      });
    } catch (error) {
      console.error("Failed to reload Python server", error);
    } finally {
      setReloading(false);
    }
  }, [reloading]);

  return (
    <div className="min-h-full w-full bg-[var(--background)] px-4 py-10 sm:px-8">
      <div className="mx-auto w-full max-w-6xl space-y-8">
        <div className="text-center space-y-3">
          <h1
            className={`${figtree.className} text-4xl font-semibold tracking-tight sm:text-5xl`}
            style={{ color: "var(--foreground)" }}
          >
            checkmate
          </h1>
          <div
            className="mx-auto h-px w-16 opacity-30"
            style={{ backgroundColor: "var(--foreground)" }}
          />
          <p
            className="text-[0.65rem] leading-relaxed sm:text-xs"
            style={{ color: "var(--foreground)", opacity: 0.7 }}
          >
            A supervised machine learning chess engine using MCTS, guided by a
            trained neural network to evaluate positions and improve move
            selection.
          </p>
        </div>
        {engineError && (
          <div className="mb-4 rounded border border-red-500 bg-red-950 px-3 py-2 text-sm text-red-200">
            {engineError}
          </div>
        )}
        {/* board scoped css */}
        <link
          rel="stylesheet"
          href="/analysis-board/css/style.css"
          type="text/css"
        />

        {/* Timer styles */}
        <style jsx>{`
          .clock.active .clock-time,
          .clock-with-selector.active .clock-time {
            font-weight: 700;
          }

          /* Ensure clock-with-selector inherits all clock styles */
          .clock-with-selector {
            /* This will inherit the existing .clock styles from style.css */
            /* We're just adding display flex to center the content */
            display: flex !important;
            align-items: center;
            justify-content: center;
          }

          .clock-selector {
            appearance: none;
            font-family: "Jost", sans-serif;
            font-size: 14px;
            font-weight: 600;
            color: var(--text-primary);
            background-color: var(--sidebar-inactive);
            padding: 4px 8px;
            border-radius: 4px;
            border: 2px solid rgba(0, 0, 0, 0.1);
            letter-spacing: 0.3px;
            cursor: pointer;
            text-align: center;
            min-width: 55px;
          }

          .clock-selector:hover {
            opacity: 0.8;
          }

          .clock-selector:focus {
            outline: none;
            opacity: 1;
          }

          .clock-selector option {
            background: var(--background);
            color: var(--foreground);
          }

          @media (max-width: 768px) {
            .clock-selector {
              font-size: 14px;
              padding: 3px 6px;
              letter-spacing: normal;
            }
          }
        `}</style>

        {/* we need jquery */}
        <Script
          src="/analysis-board/libs/jquery.3.7.1.min.js"
          strategy="afterInteractive"
          onLoad={() => {
            setScriptsLoaded((prev) => ({ ...prev, jquery: true }));
          }}
        />

        {/* chess.js is loaded automatically as a module dependency */}
        <Script
          src="/analysis-board/init-game.js"
          type="module"
          strategy="afterInteractive"
          onLoad={() => {
            setScriptsLoaded((prev) => ({ ...prev, analysis: true }));
          }}
        />

        <div
          ref={boardContainerRef}
          className="analysis-board-container dark_theme"
        >
          <main>
            <article className="container">
              <div className="main-panel">
                <div className="chess-container">
                  <div className="nameplate top">
                    <div className="profile">
                      <h4 id="black-name" className="name">
                        Black
                      </h4>
                    </div>
                    <div
                      className={`clock clock-with-selector ${
                        currentTurn === "black" && isRunning ? "active" : ""
                      }`}
                      id="black-clock"
                    >
                      {!isRunning ? (
                        <select
                          value={timeControl}
                          onChange={(e) =>
                            handleTimeControlChange(
                              Number(e.target.value) as TimeControl
                            )
                          }
                          className="clock-selector"
                        >
                          <option value={10}>10:00</option>
                          <option value={5}>5:00</option>
                          <option value={3}>3:00</option>
                          <option value={1}>1:00</option>
                        </select>
                      ) : (
                        <span className="clock-time">
                          {formatTime(blackTime)}
                        </span>
                      )}
                    </div>
                  </div>

                  <div className="chess-box">
                    <div className="eval-bar-container">
                      <div className="eval-bar">
                        <div className="eval-text"></div>
                        <div className="eval-fill"></div>
                      </div>
                    </div>
                    <div id="chessboard"></div>
                  </div>

                  <div className="nameplate bottom">
                    <div className="profile">
                      <h4 id="white-name" className="name">
                        White
                      </h4>
                    </div>
                    <div
                      className={`clock clock-with-selector ${
                        currentTurn === "white" && isRunning ? "active" : ""
                      }`}
                      id="white-clock"
                    >
                      {!isRunning ? (
                        <select
                          value={timeControl}
                          onChange={(e) =>
                            handleTimeControlChange(
                              Number(e.target.value) as TimeControl
                            )
                          }
                          className="clock-selector"
                        >
                          <option value={10}>10:00</option>
                          <option value={5}>5:00</option>
                          <option value={3}>3:00</option>
                          <option value={1}>1:00</option>
                        </select>
                      ) : (
                        <span className="clock-time">
                          {formatTime(whiteTime)}
                        </span>
                      )}
                    </div>
                  </div>
                </div>

                <div className="sidebar">
                  <div className="sidebar-header">
                    <div className="tab-buttons">
                      <button className="tab-button active" data-tab="moves">
                        Moves
                      </button>
                      <button className="tab-button" data-tab="debug">
                        Debug
                      </button>
                      <button className="tab-button" data-tab="settings">
                        Settings
                      </button>
                    </div>
                    <div className="evaluation-progress-container">
                      <div className="evaluation-progress-bar">
                        <div className="progress-bar-fill"></div>
                      </div>
                    </div>
                  </div>
                  <div className="tab-content blur-content">
                    <div id="moves-tab" className="tab-panel active">
                      <div className="top-content move-info"></div>
                      <div className="top-content engine-lines"></div>
                      <div className="moves-container">
                        <div id="move-tree" className="move-tree"></div>
                      </div>
                    </div>
                    <div id="debug-tab" className="tab-panel">
                      <div className="debug-probabilities"></div>
                      <div className="debug-logs-section"></div>
                    </div>
                    <div id="settings-tab" className="tab-panel">
                      <div className="settings-menu-container"></div>
                    </div>
                  </div>

                  <div className="bottom-content blur-content">
                    <div className="controls">
                      <button id="restart">
                        <svg
                          xmlns="http://www.w3.org/2000/svg"
                          viewBox="0 0 320 512"
                        >
                          <path
                            d="M267.5 440.6c9.5 7.9 22.8 9.7 34.1 4.4s18.4-16.6 18.4-29l0-320c0-12.4-7.2-23.7-18.4-29s-24.5-3.6-34.1 4.4l-192 160L64 241 64 96c0-17.7-14.3-32-32-32S0 78.3 0 96L0 416c0 17.7 14.3 32 32 32s32-14.3 32-32l0-145 11.5 9.6 192 160z"
                            fill="currentColor"
                          />
                        </svg>
                      </button>
                      <button id="backward">
                        <svg
                          xmlns="http://www.w3.org/2000/svg"
                          viewBox="0 0 448 512"
                        >
                          <path
                            d="M9.4 233.4c-12.5 12.5-12.5 32.8 0 45.3l160 160c12.5 12.5 32.8 12.5 45.3 0s12.5-32.8 0-45.3L109.2 288 416 288c17.7 0 32-14.3 32-32s-14.3-32-32-32l-306.7 0L214.6 118.6c12.5-12.5 12.5-32.8 0-45.3s-32.8-12.5-45.3 0l-160 160z"
                            fill="currentColor"
                          />
                        </svg>
                      </button>
                      <div className="quick-menu-container">
                        <button id="popup-quick-menu">
                          <svg
                            xmlns="http://www.w3.org/2000/svg"
                            viewBox="0 0 448 512"
                          >
                            <path
                              d="M0 96C0 78.3 14.3 64 32 64l384 0c17.7 0 32 14.3 32 32s-14.3 32-32 32L32 128C14.3 128 0 113.7 0 96zM0 256c0-17.7 14.3-32 32-32l384 0c17.7 0 32 14.3 32 32s-14.3 32-32 32L32 288c-17.7 0-32-14.3-32-32zM448 416c0 17.7-14.3 32-32 32L32 448c-17.7 0-32-14.3-32-32s14.3-32 32-32l384 0c17.7 0 32 14.3 32 32z"
                              fill="currentColor"
                            />
                          </svg>
                        </button>
                        <div className="quick-menu" id="quick-menu">
                          <div className="quick-menu-item" id="flip-board">
                            <svg
                              xmlns="http://www.w3.org/2000/svg"
                              viewBox="0 0 512 512"
                            >
                              <path
                                d="M105.1 202.6c7.7-21.8 20.2-42.3 37.8-59.8c62.5-62.5 163.8-62.5 226.3 0L386.3 160H352c-17.7 0-32 14.3-32 32s14.3 32 32 32H463.5c0 0 0 0 0 0h.4c17.7 0 32-14.3 32-32V80c0-17.7-14.3-32-32-32s-32 14.3-32 32v35.2L414.4 97.6c-87.5-87.5-229.3-87.5-316.8 0C73.2 122 55.6 150.7 44.8 181.4c-5.9 16.7 2.9 34.9 19.5 40.8s34.9-2.9 40.8-19.5zM39 289.3c-5 1.5-9.8 4.2-13.7 8.2c-4 4-6.7 8.8-8.1 14c-.3 1.2-.6 2.5-.8 3.8c-.3 1.7-.4 3.4-.4 5.1V448c0 17.7 14.3 32 32 32s32-14.3 32-32V413.3l17.6 17.5 0 0c87.5 87.4 229.3 87.4 316.7 0c24.4-24.4 42.1-53.1 52.9-83.7c5.9-16.7-2.9-34.9-19.5-40.8s-34.9 2.9-40.8 19.5c-7.7 21.8-20.2 42.3-37.8 59.8c-62.5 62.5-163.8 62.5-226.3 0L125.7 352H160c17.7 0 32-14.3 32-32s-14.3-32-32-32H48.4c-2.2 0-4.2 .9-5.6 2.3c-1.5 1.4-2.4 3.3-2.5 5.4c0 .5 0 1.1 .1 1.6z"
                                fill="currentColor"
                              />
                            </svg>
                            <span>Flip Board</span>
                          </div>
                          <div className="quick-menu-item" id="copy-pgn">
                            <svg
                              xmlns="http://www.w3.org/2000/svg"
                              viewBox="0 0 384 512"
                            >
                              <path
                                d="M192 0c-41.8 0-77.4 26.7-90.5 64L64 64C28.7 64 0 92.7 0 128L0 448c0 35.3 28.7 64 64 64l256 0c35.3 0 64-28.7 64-64l0-320c0-35.3-28.7-64-64-64l-37.5 0C269.4 26.7 233.8 0 192 0zm0 64a32 32 0 1 1 0 64 32 32 0 1 1 0-64zM72 272a24 24 0 1 1 48 0 24 24 0 1 1 -48 0zm104-16l128 0c8.8 0 16 7.2 16 16s-7.2 16-16 16l-128 0c-8.8 0-16-7.2-16-16s7.2-16 16-16zM72 368a24 24 0 1 1 48 0 24 24 0 1 1 -48 0zm88 0c0-8.8 7.2-16 16-16l128 0c8.8 0 16 7.2 16 16s-7.2 16-16 16l-128 0c-8.8 0-16-7.2-16-16z"
                                fill="currentColor"
                              />
                            </svg>
                            <span>Copy PGN</span>
                          </div>
                        </div>
                      </div>
                      <button id="forward">
                        <svg
                          xmlns="http://www.w3.org/2000/svg"
                          viewBox="0 0 448 512"
                        >
                          <path
                            d="M438.6 278.6c12.5-12.5 12.5-32.8 0-45.3l-160-160c-12.5-12.5-32.8-12.5-45.3 0s-12.5 32.8 0 45.3L338.8 224 32 224c-17.7 0-32 14.3-32 32s14.3 32 32 32l306.7 0L233.4 393.4c-12.5 12.5-12.5 32.8 0 45.3s32.8 12.5 45.3 0l160-160z"
                            fill="currentColor"
                          />
                        </svg>
                      </button>
                      <button id="skip-to-end">
                        <svg
                          xmlns="http://www.w3.org/2000/svg"
                          viewBox="0 0 320 512"
                        >
                          <path
                            d="M52.5 440.6c-9.5 7.9-22.8 9.7-34.1 4.4S0 428.4 0 416L0 96C0 83.6 7.2 72.3 18.4 67s24.5-3.6 34.1 4.4l192 160L256 241l0-145c0-17.7 14.3-32 32-32s32 14.3 32 32l0 320c0 17.7-14.3 32-32 32s-32-14.3-32-32l0-145-11.5 9.6-192 160z"
                            fill="currentColor"
                          />
                        </svg>
                      </button>
                    </div>
                  </div>

                  <div className="analysis-overlay active">
                    <div className="analysis-content">
                      <h2>Analyzing Game</h2>
                      <p>Stockfish is analyzing your game...</p>
                      <div className="analysis-progress">
                        <div className="analysis-progress-bar"></div>
                      </div>
                      <div className="fun-fact">
                        Here&apos;s a fun fact: the website is still loading!
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </article>
          </main>
        </div>
      </div>
      {gameResult && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 px-4 py-12">
          <div className="w-full max-w-md rounded-3xl border border-white/10 bg-[var(--background)] p-8 text-center shadow-2xl">
            <p
              className="text-[0.6rem] uppercase tracking-[0.3em]"
              style={{ color: "var(--foreground)", opacity: 0.6 }}
            >
              {gameResult.reason}
            </p>
            <h2
              className={`${figtree.className} mt-4 text-3xl font-semibold sm:text-4xl`}
              style={{ color: "var(--foreground)" }}
            >
              {gameResult.outcome === "win" ? "You win" : "You lose"}
            </h2>
            <p
              className="mt-3 text-sm leading-relaxed"
              style={{ color: "var(--foreground)", opacity: 0.7 }}
            >
              {gameResult.outcome === "win"
                ? "Beautiful mate. Spin it up again to see what the bot tries next."
                : "The bot found mate this time. Hit retry and go for the comeback."}
            </p>
            <button
              className="mt-6 inline-flex w-full items-center justify-center rounded-full border border-[var(--foreground)] px-4 py-3 text-sm font-semibold uppercase tracking-[0.3em]"
              style={{ color: "var(--foreground)" }}
              onClick={handleRetry}
            >
              Play again
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
