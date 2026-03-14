import { useCallback, useEffect, useRef, useState } from "react";
import { apiClient } from "../services/apiClient";
import type { VerificationResult } from "../App";
import type { AnalyticsSnapshot } from "../components/BehaviorDashboard";

type BehaviorCollectorOptions = {
  sessionId: string;
  onVerified: (result: VerificationResult) => void;
};

type BrowserMetadata = {
  userAgent: string;
  language?: string;
  platform?: string;
  screenWidth?: number;
  screenHeight?: number;
  webglFingerprint?: string;
  canvasFingerprint?: string;
  deviceEntropy?: number;
  webdriver?: boolean;
  touchPoints?: number;
};

type BehaviorBatchPayload = {
  session_id: string;
  started_at: number;
  ended_at: number;
  mouse_moves: { timestamp: number; position: { x: number; y: number } }[];
  scrolls: { timestamp: number; delta_y: number }[];
  clicks: { timestamp: number; button: string }[];
  key_presses: { timestamp: number; key: string }[];
  focus_events: { timestamp: number; focused: boolean }[];
  metadata?: BrowserMetadata;
};

type VerifyResponseDto = {
  session_id: string;
  human_probability: number;
  risk_level: "low" | "medium" | "high";
  recommended_action: string;
  risk_score: number;
};

type AnalyticsResponseDto = {
  total_sessions: number;
  average_human_probability: number;
  risk_distribution: { label: string; count: number }[];
};

const computeCanvasFingerprint = (): string => {
  try {
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    if (!ctx) return "none";
    ctx.textBaseline = "top";
    ctx.font = "14px 'Arial'";
    ctx.fillText("botguard-ai-fingerprint", 2, 2);
    return canvas.toDataURL();
  } catch {
    return "error";
  }
};

const computeWebGLFingerprint = (): string => {
  try {
    const canvas = document.createElement("canvas");
    const gl =
      canvas.getContext("webgl") || canvas.getContext("experimental-webgl");
    if (!gl) return "none";
    const rendererInfo = (gl as any).getExtension("WEBGL_debug_renderer_info");
    if (rendererInfo) {
      const vendor = gl.getParameter(rendererInfo.UNMASKED_VENDOR_WEBGL);
      const renderer = gl.getParameter(rendererInfo.UNMASKED_RENDERER_WEBGL);
      return `${vendor}::${renderer}`;
    }
    return "webgl";
  } catch {
    return "error";
  }
};

const computeDeviceEntropy = (): number => {
  const parts: (string | number | undefined)[] = [
    navigator.userAgent,
    navigator.language,
    navigator.platform,
    window.innerWidth,
    window.innerHeight,
    (navigator as any).hardwareConcurrency,
    (navigator as any).deviceMemory
  ];
  const str = parts.join("|");
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = (hash << 5) - hash + str.charCodeAt(i);
    hash |= 0;
  }
  return Math.abs(hash);
};

export const useBehaviorCollector = (options: BehaviorCollectorOptions) => {
  const { sessionId, onVerified } = options;

  const [analytics, setAnalytics] = useState<AnalyticsSnapshot | null>(null);

  const batchRef = useRef<BehaviorBatchPayload | null>(null);
  const intervalIdRef = useRef<number | null>(null);

  const initBatch = useCallback(() => {
    const now = performance.now();
    const metadata: BrowserMetadata = {
      userAgent: navigator.userAgent,
      language: navigator.language,
      platform: navigator.platform,
      screenWidth: window.innerWidth,
      screenHeight: window.innerHeight,
      webglFingerprint: computeWebGLFingerprint(),
      canvasFingerprint: computeCanvasFingerprint(),
      deviceEntropy: computeDeviceEntropy(),
      webdriver: (navigator as any).webdriver === true,
      touchPoints: (navigator as any).maxTouchPoints ?? 0
    };
    batchRef.current = {
      session_id: sessionId,
      started_at: now,
      ended_at: now,
      mouse_moves: [],
      scrolls: [],
      clicks: [],
      key_presses: [],
      focus_events: [],
      metadata
    };
  }, [sessionId]);

  const flushBatch = useCallback(async () => {
    if (!batchRef.current) return;
    const payload = batchRef.current;
    if (
      payload.mouse_moves.length === 0 &&
      payload.scrolls.length === 0 &&
      payload.clicks.length === 0 &&
      payload.key_presses.length === 0 &&
      payload.focus_events.length === 0
    ) {
      return;
    }

    try {
      await apiClient.post("/collect-behavior", payload);
      initBatch();
    } catch (err) {
      // For prototype we only log to console; production would have monitoring hooks.
      console.error("Failed to send behavior batch", err);
    }
  }, [initBatch]);

  const attachListeners = useCallback(() => {
    if (!batchRef.current) {
      initBatch();
    }

    const handleMouseMove = (event: MouseEvent) => {
      if (!batchRef.current) return;
      const now = performance.now();
      batchRef.current.mouse_moves.push({
        timestamp: now,
        position: { x: event.clientX, y: event.clientY }
      });
      batchRef.current.ended_at = now;
    };

    const handleScroll = () => {
      if (!batchRef.current) return;
      const now = performance.now();
      batchRef.current.scrolls.push({
        timestamp: now,
        delta_y: window.scrollY
      });
      batchRef.current.ended_at = now;
    };

    const handleClick = (event: MouseEvent) => {
      if (!batchRef.current) return;
      const now = performance.now();
      const button =
        event.button === 0 ? "left" : event.button === 1 ? "middle" : "right";
      batchRef.current.clicks.push({
        timestamp: now,
        button
      });
      batchRef.current.ended_at = now;
    };

    const handleKeyDown = (event: KeyboardEvent) => {
      if (!batchRef.current) return;
      const now = performance.now();
      batchRef.current.key_presses.push({
        timestamp: now,
        key: event.key
      });
      batchRef.current.ended_at = now;
    };

    const handleFocus = () => {
      if (!batchRef.current) return;
      const now = performance.now();
      batchRef.current.focus_events.push({
        timestamp: now,
        focused: true
      });
      batchRef.current.ended_at = now;
    };

    const handleBlur = () => {
      if (!batchRef.current) return;
      const now = performance.now();
      batchRef.current.focus_events.push({
        timestamp: now,
        focused: false
      });
      batchRef.current.ended_at = now;
    };

    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("scroll", handleScroll);
    window.addEventListener("click", handleClick);
    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("focus", handleFocus);
    window.addEventListener("blur", handleBlur);

    intervalIdRef.current = window.setInterval(() => {
      void flushBatch();
      void refreshAnalytics();
    }, 4000);

    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("scroll", handleScroll);
      window.removeEventListener("click", handleClick);
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("focus", handleFocus);
      window.removeEventListener("blur", handleBlur);
      if (intervalIdRef.current !== null) {
        window.clearInterval(intervalIdRef.current);
      }
    };
  }, [flushBatch]);

  const refreshAnalytics = useCallback(async () => {
    try {
      const res = await apiClient.get<AnalyticsResponseDto>("/analytics");
      setAnalytics({
        totalSessions: res.data.total_sessions,
        averageHumanProbability: res.data.average_human_probability,
        riskBuckets: res.data.risk_distribution
      });
    } catch (err) {
      console.error("Failed to fetch analytics", err);
    }
  }, []);

  const verifySession = useCallback(async () => {
    await flushBatch();
    try {
      const latest = batchRef.current;
      const payload = {
        session_id: sessionId,
        latest_batch: latest
          ? {
              ...latest
            }
          : null
      };
      const res = await apiClient.post<VerifyResponseDto>("/verify-session", payload);
      const result: VerificationResult = {
        sessionId: res.data.session_id,
        humanProbability: res.data.human_probability,
        riskLevel: res.data.risk_level,
        recommendedAction: res.data.recommended_action,
        riskScore: res.data.risk_score
      };
      onVerified(result);
      await refreshAnalytics();
    } catch (err) {
      console.error("Failed to verify session", err);
    }
  }, [flushBatch, onVerified, refreshAnalytics, sessionId]);

  useEffect(() => {
    const cleanup = attachListeners();
    void refreshAnalytics();
    return () => {
      cleanup?.();
    };
  }, [attachListeners, refreshAnalytics]);

  return {
    latestBatch: batchRef.current,
    analytics,
    verifySession
  };
};

