import React from "react";
import type { VerificationResult } from "../App";

type Props = {
  sessionId: string;
  verification: VerificationResult | null;
};

const VerificationStatus: React.FC<Props> = ({ sessionId, verification }) => {
  const statusLabel =
    verification?.riskLevel === "low"
      ? "Likely Human"
      : verification?.riskLevel === "medium"
      ? "Uncertain – Consider Challenge"
      : verification?.riskLevel === "high"
      ? "Likely Bot"
      : "Not yet evaluated";

  const statusColor =
    verification?.riskLevel === "low"
      ? "text-emerald-400"
      : verification?.riskLevel === "medium"
      ? "text-amber-300"
      : verification?.riskLevel === "high"
      ? "text-rose-400"
      : "text-slate-400";

  const probabilityPercent = verification
    ? Math.round(verification.humanProbability * 100)
    : null;
  const riskScore = verification?.riskScore;

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4 shadow-sm space-y-3">
      <h2 className="text-lg font-semibold text-slate-50">Human Verification Status</h2>
      <div className="space-y-1">
        <p className={`text-sm font-medium ${statusColor}`}>{statusLabel}</p>
        <p className="text-[11px] text-slate-500 break-all">
          Session ID: <span className="font-mono">{sessionId}</span>
        </p>
      </div>
      <div className="mt-2">
        <p className="text-xs text-slate-400 mb-1">Human confidence score</p>
        <div className="h-2 w-full rounded-full bg-slate-800 overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-rose-500 via-amber-400 to-emerald-400 transition-all duration-500"
            style={{ width: `${probabilityPercent ?? 0}%` }}
          />
        </div>
        <p className="mt-1 text-xs text-slate-400">
          {probabilityPercent !== null ? `${probabilityPercent}%` : "Awaiting verification"}
        </p>
      </div>
      {riskScore !== undefined && (
        <div className="mt-2">
          <p className="text-xs text-slate-400 mb-1">Composite risk score</p>
          <div className="h-2 w-full rounded-full bg-slate-800 overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-emerald-500 via-amber-400 to-rose-500 transition-all duration-500"
              style={{ width: `${riskScore}%` }}
            />
          </div>
          <p className="mt-1 text-xs text-slate-400">{Math.round(riskScore)}/100</p>
        </div>
      )}
      {verification && (
        <div className="mt-2 rounded-md border border-slate-700 bg-slate-900/60 px-3 py-2 text-xs text-slate-300">
          <p>
            <span className="font-semibold text-slate-100">Recommended action: </span>
            <span className="uppercase tracking-wide text-[11px]">
              {verification.recommendedAction}
            </span>
          </p>
          <p className="mt-1 text-slate-400">
            This is a passive behavioral decision; combine with your normal auth and risk
            engine in production.
          </p>
        </div>
      )}
    </div>
  );
};

export default VerificationStatus;

