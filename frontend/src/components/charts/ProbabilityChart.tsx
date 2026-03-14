import React from "react";
import { Doughnut } from "react-chartjs-2";
import {
  ArcElement,
  Chart as ChartJS,
  Legend,
  Tooltip
} from "chart.js";
import type { VerificationResult } from "../../App";

ChartJS.register(ArcElement, Tooltip, Legend);

type Props = {
  verification: VerificationResult | null;
};

const ProbabilityChart: React.FC<Props> = ({ verification }) => {
  const humanProb = verification?.humanProbability ?? 0;

  const data = {
    labels: ["Human", "Bot"],
    datasets: [
      {
        data: [humanProb, 1 - humanProb],
        backgroundColor: ["#22c55e", "#f97373"],
        borderColor: ["#16a34a", "#fb7185"],
        borderWidth: 1
      }
    ]
  };

  const options = {
    responsive: true,
    plugins: {
      legend: {
        display: true,
        labels: {
          color: "#cbd5f5",
          font: { size: 10 }
        }
      }
    }
  };

  return (
    <div className="h-48 flex items-center justify-center">
      <Doughnut data={data} options={options} />
    </div>
  );
};

export default ProbabilityChart;

