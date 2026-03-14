import React, { useState } from "react";
import LoginPage from "./components/LoginPage";
import VerificationStatus from "./components/VerificationStatus";
import BehaviorDashboard from "./components/BehaviorDashboard";
import { useBehaviorCollector } from "./hooks/useBehaviorCollector";

export type VerificationResult = {
  sessionId: string;
  humanProbability: number;
  riskLevel: "low" | "medium" | "high";
  recommendedAction: string;
  riskScore: number;
};

const App: React.FC = () => {
  const [sessionId, setSessionId] = useState<string>(() => crypto.randomUUID());
  const [verification, setVerification] = useState<VerificationResult | null>(null);

  const { latestBatch, analytics, verifySession } = useBehaviorCollector({
    sessionId,
    onVerified: setVerification
  });

  const handleNewSession = () => {
    setSessionId(crypto.randomUUID());
    setVerification(null);
  };

  const handleVerifyClick = async () => {
    await verifySession();
  };

  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b border-slate-800 bg-slate-950/80 backdrop-blur">
        <div className="max-w-6xl mx-auto px-4 py-4 flex items-center justify-between">
          <div>
            <h1 className="text-xl font-semibold text-slate-50">BotGuard AI</h1>
            <p className="text-sm text-slate-400">
              Passive ML-based human verification as a CAPTCHA alternative
            </p>
          </div>
          <button
            onClick={handleNewSession}
            className="px-3 py-1.5 rounded-md bg-slate-800 text-slate-100 text-sm hover:bg-slate-700 border border-slate-700"
          >
            New Session
          </button>
        </div>
      </header>

      <main className="flex-1 max-w-6xl mx-auto w-full px-4 py-6 grid gap-6 md:grid-cols-5">
        <section className="md:col-span-2 space-y-4">
          <LoginPage onVerifyClick={handleVerifyClick} />
          <VerificationStatus sessionId={sessionId} verification={verification} />
        </section>

        <section className="md:col-span-3">
          <BehaviorDashboard analytics={analytics} verification={verification} />
        </section>
      </main>

      <footer className="border-t border-slate-800 py-4 text-center text-xs text-slate-500">
        BotGuard AI prototype – for demonstration purposes only.
      </footer>
    </div>
  );
};

export default App;

