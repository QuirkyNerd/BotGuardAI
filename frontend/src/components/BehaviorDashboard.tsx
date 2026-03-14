import React from "react";
import type { VerificationResult } from "../App";
import ProbabilityChart from "./charts/ProbabilityChart";
import RiskDistributionChart from "./charts/RiskDistributionChart";

export type AnalyticsSnapshot = {
  totalSessions: number;
  averageHumanProbability: number;
  riskBuckets: { label: string; count: number }[];
};

type Props = {
  analytics: AnalyticsSnapshot | null;
  verification: VerificationResult | null;
};

const BehaviorDashboard: React.FC<Props> = ({ analytics, verification }) => {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4 shadow-sm space-y-4 h-full">
      <div className="flex items-center justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold text-slate-50">Behavior Analytics</h2>
          <p className="text-xs text-slate-400">
            Real-time view of session verification outcomes and risk distribution.
          </p>
        </div>
        <div className="text-right">
          <p className="text-xs text-slate-500">Total evaluated sessions</p>
          <p className="text-lg font-semibold text-slate-50">
            {analytics?.totalSessions ?? 0}
          </p>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <div className="rounded-lg border border-slate-800 bg-slate-950/50 p-3">
          <h3 className="text-xs font-semibold text-slate-300 mb-2">
            Latest Session Probability
          </h3>
          <ProbabilityChart verification={verification} />
        </div>

        <div className="rounded-lg border border-slate-800 bg-slate-950/50 p-3">
          <h3 className="text-xs font-semibold text-slate-300 mb-2">
            Risk Distribution (All Sessions)
          </h3>
          <RiskDistributionChart analytics={analytics} />
        </div>
      </div>
    </div>
  );
};

export default BehaviorDashboard;

