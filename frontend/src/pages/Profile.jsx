import React, { useEffect, useState } from "react";
import { Save, UserRound } from "lucide-react";
import { Button, Card } from "../components/ui";
import { useAuthStore } from "../store/authStore";
export default function Profile() {
    const user = useAuthStore((s) => s.user);
    const updateProfile = useAuthStore((s) => s.updateProfile);
    const loading = useAuthStore((s) => s.loading);
    const [username, setUsername] = useState(user?.username || "");
    const [email, setEmail] = useState(user?.email || "");
    const [saved, setSaved] = useState(false);
    useEffect(() => {
        setUsername(user?.username || "");
        setEmail(user?.email || "");
    }, [user]);
    const submit = async (event) => {
        event.preventDefault();
        await updateProfile({ username, email });
        setSaved(true);
        setTimeout(() => setSaved(false), 1800);
    };
    return (<div className="p-5">
      <Card className="mx-auto max-w-2xl p-6">
        <div className="mb-6 flex items-center gap-3">
          <div className="profile-user-icon-tile grid h-12 w-12 place-items-center rounded-xl bg-violet-950 text-violet-300"><UserRound className="profile-user-icon" /></div>
          <div>
            <h2 className="text-base font-semibold text-white">Account Profile</h2>
            <p className="text-xs text-slate-500">Update profile fields supported by the backend.</p>
          </div>
        </div>
        <form className="space-y-4" onSubmit={submit}>
          <div><label className="label">Full Name</label><input className="field" value={username} onChange={(e) => setUsername(e.target.value)}/></div>
          <div><label className="label">Email Address</label><input className="field" type="email" value={email} onChange={(e) => setEmail(e.target.value)}/></div>
          <Button loading={loading} type="submit"><Save className="h-4 w-4"/> Save Profile</Button>
          {saved && <span className="ml-3 text-xs text-emerald-400">Profile saved.</span>}
        </form>
      </Card>
    </div>);
}
