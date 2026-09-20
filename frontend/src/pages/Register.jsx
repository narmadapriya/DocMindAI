import React, {
  useMemo,
  useState,
} from "react";

import {
  ArrowRight,
  Check,
  LockKeyhole,
  Mail,
  UserRound,
  X,
} from "lucide-react";

import {
  Link,
  useNavigate,
} from "react-router-dom";

import {
  Logo,
} from "../components/Logo";

import {
  Button,
  Card,
} from "../components/ui";

import {
  PasswordVisibilityToggle,
} from "../components/PasswordVisibilityToggle";

import {
  useAuthStore,
} from "../store/authStore";

function passwordRules(password) {
  return {
    length:
      password.length >= 8,
    upper:
      /[A-Z]/.test(password),
    lower:
      /[a-z]/.test(password),
    number:
      /\d/.test(password),
    special:
      /[^A-Za-z0-9]/.test(
        password,
      ),
  };
}

const RULES = [
  [
    "length",
    "Minimum 8 characters",
  ],
  [
    "upper",
    "Uppercase letter (A–Z)",
  ],
  [
    "lower",
    "Lowercase letter (a–z)",
  ],
  [
    "number",
    "Number (0–9)",
  ],
  [
    "special",
    "Special character (!, @, #, $, %, ^, &, *)",
  ],
];

export default function Register() {
  const navigate =
    useNavigate();

  const register =
    useAuthStore(
      (state) =>
        state.register,
    );

  const loading =
    useAuthStore(
      (state) =>
        state.loading,
    );

  const error =
    useAuthStore(
      (state) =>
        state.error,
    );

  const [
    form,
    setForm,
  ] = useState({
    username: "",
    email: "",
    password: "",
    confirmPassword: "",
  });

  const [
    visible,
    setVisible,
  ] = useState({
    password: false,
    confirm: false,
  });

  const checks =
    useMemo(
      () =>
        passwordRules(
          form.password,
        ),
      [form.password],
    );

  const rulesPassed =
    Object.values(
      checks,
    ).filter(Boolean).length;

  const allRulesPassed =
    rulesPassed ===
    RULES.length;

  const passwordsMatch =
    form.confirmPassword
      .length > 0 &&
    form.password ===
      form.confirmPassword;

  const formReady =
    form.username.trim() &&
    form.email.trim() &&
    allRulesPassed &&
    passwordsMatch;

  const clearRegisterForm =
    () => {
      setForm({
        username: "",
        email: "",
        password: "",
        confirmPassword: "",
      });
      setVisible({
        password: false,
        confirm: false,
      });
      useAuthStore.setState({
        error: null,
      });
    };


  const submit =
    async (event) => {
      event.preventDefault();

      if (!formReady) {
        return;
      }

      try {
        await register({
          username:
            form.username.trim(),
          email:
            form.email
              .trim()
              .toLowerCase(),
          password:
            form.password,
        });

        navigate(
          "/login",
          {
            replace: true,
          },
        );
      } catch {
        // authStore owns backend error state.
      }
    };

  return (
    <div
      className="
        auth-page
        grid
        min-h-screen
        place-items-center
        px-4
        py-8
      "
    >
      <div
        className="
          w-full
          max-w-[470px]
        "
      >
        <div
          className="
            mb-5
            flex
            flex-col
            items-center
            text-center
          "
        >
          <Logo compact />

          <h1
            className="
              mt-3
              text-lg
              font-bold
              text-white
            "
          >
            DocMind AI Gateway
          </h1>

          <p
            className="
              mt-1
              text-[11px]
              text-slate-500
            "
          >
            Join free and
            experience
            multi-document
            semantic reasoning
            instantly.
          </p>
        </div>

        <Card
          className="
            auth-card
            p-5
            sm:p-6
          "
        >
          <h2
            className="
              text-base
              font-bold
              text-white
            "
          >
            Create Account
          </h2>

          <p
            className="
              mt-1
              text-[11px]
              text-slate-500
            "
          >
            Sign up for local
            sandbox access and
            document indexing.
          </p>

          <form
            className="
              mt-5
              space-y-4
            "
            onSubmit={submit}
          >
            <div>
              <label
                className="label"
              >
                Full Name
              </label>

              <div
                className="
                  relative
                "
              >
                <UserRound
                  className="
                    absolute
                    left-3
                    top-3
                    h-4
                    w-4
                    text-slate-500
                  "
                />

                <input
                  className="
                    field
                    pl-9
                  "
                  required
                  autoComplete="name"
                  value={
                    form.username
                  }
                  onChange={(
                    event,
                  ) =>
                    setForm({
                      ...form,
                      username:
                        event.target
                          .value,
                    })
                  }
                  placeholder="John Doe"
                />
              </div>
            </div>

            <div>
              <label
                className="label"
              >
                Email Address
              </label>

              <div
                className="
                  relative
                "
              >
                <Mail
                  className="
                    absolute
                    left-3
                    top-3
                    h-4
                    w-4
                    text-slate-500
                  "
                />

                <input
                  className="
                    field
                    pl-9
                  "
                  type="email"
                  autoComplete="username"
                  required
                  value={
                    form.email
                  }
                  onChange={(
                    event,
                  ) =>
                    setForm({
                      ...form,
                      email:
                        event.target
                          .value,
                    })
                  }
                  placeholder="john@example.com"
                />
              </div>
            </div>

            <div>
              <label
                className="label"
              >
                Password
              </label>

              <div
                className="
                  relative
                "
              >
                <LockKeyhole
                  className="
                    absolute
                    left-3
                    top-3
                    h-4
                    w-4
                    text-slate-500
                  "
                />

                <input
                  className="
                    field
                    pl-9
                    pr-11
                  "
                  type={
                    visible.password
                      ? "text"
                      : "password"
                  }
                  autoComplete="new-password"
                  required
                  value={
                    form.password
                  }
                  onChange={(
                    event,
                  ) =>
                    setForm({
                      ...form,
                      password:
                        event.target
                          .value,
                    })
                  }
                  placeholder="Create strong password"
                />

                <PasswordVisibilityToggle
                  visible={
                    visible.password
                  }
                  onToggle={() =>
                    setVisible({
                      ...visible,
                      password:
                        !visible.password,
                    })
                  }
                />
              </div>
            </div>

            <div
              className="
                rounded-xl
                border
                border-violet-800/60
                bg-violet-950/15
                p-3
                sm:p-4
              "
            >
              <div
                className="
                  mb-3
                  flex
                  items-center
                  justify-between
                  gap-3
                  text-[11px]
                  font-semibold
                "
              >
                <span>
                  The 8-4 Password Rule
                </span>

                <span
                  className={`
                    rounded-md
                    px-2
                    py-1
                    ${
                      allRulesPassed
                        ? "password-rules-badge-complete bg-violet-950/50 text-violet-300"
                        : "bg-amber-950/50 text-amber-300"
                    }
                  `}
                >
                  {rulesPassed}/5
                  RULES MET
                </span>
              </div>

              <div
                className="
                  grid
                  gap-2
                  sm:grid-cols-2
                "
              >
                {RULES.map(
                  ([
                    key,
                    label,
                  ]) => {
                    const passed =
                      checks[key];

                    return (
                      <div
                        key={key}
                        className={`
                          flex
                          items-center
                          gap-2
                          rounded-md
                          border
                          px-2.5
                          py-2
                          text-[11px]
                          ${
                            passed
                              ? "password-rule-success border-emerald-900/70 bg-emerald-950/15 text-emerald-400"
                              : "border-slate-800 text-slate-500"
                          }
                        `}
                      >
                        {passed ? (
                          <Check
                            className="
                              h-3
                              w-3
                              shrink-0
                            "
                          />
                        ) : (
                          <X
                            className="
                              h-3
                              w-3
                              shrink-0
                            "
                          />
                        )}

                        <span>
                          {label}
                        </span>
                      </div>
                    );
                  },
                )}
              </div>
            </div>

            <div>
              <label
                className="label"
              >
                Confirm Password
              </label>

              <div
                className="
                  relative
                "
              >
                <LockKeyhole
                  className="
                    absolute
                    left-3
                    top-3
                    h-4
                    w-4
                    text-slate-500
                  "
                />

                <input
                  className={`
                    field
                    pl-9
                    pr-11
                    ${
                      form.confirmPassword &&
                      !passwordsMatch
                        ? "border-rose-700"
                        : ""
                    }
                  `}
                  type={
                    visible.confirm
                      ? "text"
                      : "password"
                  }
                  autoComplete="new-password"
                  required
                  value={
                    form.confirmPassword
                  }
                  onChange={(
                    event,
                  ) =>
                    setForm({
                      ...form,
                      confirmPassword:
                        event.target
                          .value,
                    })
                  }
                  placeholder="Confirm password"
                />

                <PasswordVisibilityToggle
                  visible={
                    visible.confirm
                  }
                  onToggle={() =>
                    setVisible({
                      ...visible,
                      confirm:
                        !visible.confirm,
                    })
                  }
                />
              </div>

              {form.confirmPassword && (
                <p
                  className={`
                    mt-1.5
                    text-[10px]
                    ${
                      passwordsMatch
                        ? "text-emerald-400"
                        : "text-rose-400"
                    }
                  `}
                >
                  {passwordsMatch
                    ? "Passwords match."
                    : "Passwords do not match."}
                </p>
              )}
            </div>

            {error && (
              <div
                className="
                  rounded-lg
                  border
                  border-rose-900
                  bg-rose-950/20
                  p-2.5
                  text-xs
                  text-rose-300
                "
              >
                {error}
              </div>
            )}

            <div className="flex gap-2">
              <Button
                type="submit"
                loading={loading}
                disabled={!formReady}
                className="flex-1"
              >
                Register Account
                <ArrowRight
                  className="
                    h-4
                    w-4
                  "
                />
              </Button>

              <Button
                type="button"
                variant="secondary"
                disabled={loading}
                onClick={clearRegisterForm}
                className="px-3 text-[11px]"
              >
                Clear
              </Button>
            </div>
          </form>

          <p
            className="
              mt-6
              text-center
              text-[11px]
              text-slate-400
            "
          >
            Already have an
            account?{" "}
            <Link
              className="
                font-semibold
                text-violet-400
              "
              to="/login"
            >
              Log In
            </Link>
          </p>
        </Card>
      </div>
    </div>
  );
}
