import React from "react";

type Props = {
  onVerifyClick: () => void;
};

const LoginPage: React.FC<Props> = ({ onVerifyClick }) => {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4 shadow-sm space-y-4">
      <h2 className="text-lg font-semibold text-slate-50">Login Simulation</h2>
      <p className="text-xs text-slate-400">
        Interact with the page naturally (move mouse, scroll, type) and then click
        &quot;Verify as Human&quot;. The system will analyze your behavior passively.
      </p>
      <div className="space-y-3">
        <div>
          <label className="block text-xs text-slate-400 mb-1">Email</label>
          <input
            type="email"
            placeholder="you@example.com"
            className="w-full rounded-md bg-slate-950 border border-slate-700 px-3 py-2 text-sm text-slate-50 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
          />
        </div>
        <div>
          <label className="block text-xs text-slate-400 mb-1">Password</label>
          <input
            type="password"
            placeholder="••••••••"
            className="w-full rounded-md bg-slate-950 border border-slate-700 px-3 py-2 text-sm text-slate-50 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
          />
        </div>
      </div>
      <div className="flex items-center justify-between pt-2">
        <button
          type="button"
          onClick={onVerifyClick}
          className="inline-flex items-center justify-center rounded-md bg-emerald-500 px-4 py-2 text-sm font-medium text-emerald-950 shadow-sm hover:bg-emerald-400 focus:outline-none focus:ring-2 focus:ring-emerald-500"
        >
          Verify as Human
        </button>
        <span className="text-[11px] text-slate-500">
          No CAPTCHA. Behavior only.
        </span>
      </div>
    </div>
  );
};

export default LoginPage;

