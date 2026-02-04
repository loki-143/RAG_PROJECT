import { createRoot } from "react-dom/client";
import App from "./App.jsx";
import "./index.css";

// Ensure dark mode is applied on load
document.documentElement.classList.add('dark');

createRoot(document.getElementById("root")).render(<App />);
