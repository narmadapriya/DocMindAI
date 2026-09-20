import React from "react";
import { Eye, EyeOff } from "lucide-react";

export function PasswordVisibilityToggle({ visible, onToggle, label = "password" }) {
  return (
    <button
      type="button"
      className="password-toggle"
      onClick={onToggle}
      aria-label={`${visible ? "Hide" : "Show"} ${label}`}
      title={`${visible ? "Hide" : "Show"} ${label}`}
    >
      {visible ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
    </button>
  );
}
