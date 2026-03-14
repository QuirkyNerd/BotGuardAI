import React from "react";
import {
  BarElement,
  CategoryScale,
  Chart as ChartJS,
  Legend,
  LinearScale,
  Tooltip
} from "chart.js";
import { Bar } from "react-chartjs-2";
import type { AnalyticsSnapshot } from "../BehaviorDashboard";

ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip, Legend);

type Props = {
  analytics: AnalyticsSnapshot | null;
};

const RiskDistributionChart: React.FC<Props> = ({ analytics }) => {
  const buckets = analytics?.riskBuckets ?? [];

  const labels = buckets.length ? buckets.map((b) => b.label) : ["low", "medium", "high"];
  const counts =
    buckets.length > 0
      ? buckets.map((b) => b.count)
      : [0, 0, 0];

  const data = {
    labels,
    datasets: [
      {
        label: "Sessions",
        data: counts,
        backgroundColor: ["#22c55e", "#facc15", "#f97373"]
      }
    ]
  };

  const options = {
    responsive: true,
    plugins: {
      legend: {
        display: false
      }
    },
    scales: {
      x: {
        ticks: {
          color: "#94a3b8",
          font: { size: 10 }
        }
      },
      y: {
        ticks: {
          color: "#94a3b8",
          font: { size: 10 },
          precision: 0
        }
      }
    }
  };

  return (
    <div className="h-48">
      <Bar data={data} options={options} />
    </div>
  );
};

export default RiskDistributionChart;

