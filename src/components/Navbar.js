

import React from 'react';
import { Link } from 'react-router-dom';
import './navbar.css'; // Import the CSS file
import '../styles/navbar.css';


export default function Navbar() {
  return (
    <nav className="navbar">
      <img src="/logo.png" alt="Logo" className="navbar-logo"/>
      <div className="navbar-links">
        <Link to="/">Home</Link>
        <Link to="/features">Features</Link>
        <Link to="/pricing">Pricing</Link>
        <Link to="/signin">Sign In</Link>
        <Link to="/signup">Sign Up</Link>
      </div>
    </nav>
  );
}

