import React from 'react';
import './sidebar.css';

const tiers = [
  {
    name: 'Basic (Starter)',
    idealUser: 'Students, freelancers, small startups',
    coreValue: 'Quick charting and insights — lightweight AI assistant',
    color: '#15d4fa'
  },
  {
    name: 'Intermediate (Professional)',
    idealUser: 'MSMEs, analysts, small data teams',
    coreValue: 'Full analytics suite with AI recommendations and multi-dataset dashboards',
    color: '#28c87e'
  },
  {
    name: 'Pro (Enterprise / AI Copilot)',
    idealUser: 'Enterprises, consulting firms, advanced analysts',
    coreValue: 'Predictive, prescriptive, and generative AI-driven analytics with collaboration, security, and integration capabilities',
    color: '#ea5758'
  }
];

export default function Sidebar() {
  return (
    <aside className="sidebar">
      <h2 style={{marginBottom: 24}}>Subscription Level</h2>
      {tiers.map(tier => (
        <div className="sidebar-tier-card" key={tier.name} style={{borderLeft: `8px solid ${tier.color}`}}>
          <div className="sidebar-tier-name">{tier.name}</div>
          <div className="sidebar-tier-section">
            <span className="sidebar-label">Ideal User:</span> {tier.idealUser}
          </div>
          <div className="sidebar-tier-section">
            <span className="sidebar-label">Core Value:</span> {tier.coreValue}
          </div>
        </div>
      ))}
    </aside>
  );
}
