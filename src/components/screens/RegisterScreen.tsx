import React, { useState } from 'react';
import { AppScreen } from '../../types';
import { ASSETS } from '../../data/mockData';
import { GsapTextHighlight } from '../GsapTextHighlight';
import { api } from '../../services/backend';

interface RegisterScreenProps {
  onNavigate: (screen: AppScreen) => void;
  onShowToast: (msg: string, icon?: string) => void;
}

export const RegisterScreen: React.FC<RegisterScreenProps> = ({
  onNavigate,
  onShowToast,
}) => {
  const [role, setRole] = useState<'traveller' | 'collaborator'>('traveller');
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [phone, setPhone] = useState('');
  const [agreedToTerms, setAgreedToTerms] = useState(false);

  // Collaborator specific fields
  const [hostName, setHostName] = useState('');
  const [businessType, setBusinessType] = useState('');

  const handleRegister = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!agreedToTerms) {
      onShowToast('Please accept the Community Safety Guidelines & Terms', 'warning');
      return;
    }
    setIsSubmitting(true);
    try {
      const result = await api.register({
        role: role === 'traveller' ? 'user' : 'collaborator', name: name.trim(), email: email.trim(), password,
        phone: phone || undefined,
        ...(role === 'collaborator' ? { collaborator: { business_name: hostName.trim(), business_type: businessType } } : {}),
      });
      onShowToast(role === 'traveller' ? `Welcome to SideQuest, ${result.user.name}.` : 'Host account created. Upload your verification document next.', role === 'traveller' ? 'verified' : 'storefront');
      onNavigate(result.user.role === 'admin' ? 'admin' : result.user.role === 'user' ? 'preferences' : 'collaborator');
    } catch (error) {
      onShowToast(error instanceof Error ? error.message : 'Unable to create your account.', 'error');
    } finally { setIsSubmitting(false); }
  };

  return (
    <div className="flex flex-col w-full max-w-md md:max-w-2xl mx-auto px-4 pt-4 pb-24 space-y-5">
      {/* Header */}
      <div className="text-center space-y-1.5">
        <img
          alt="SideQuest Logo"
          src={ASSETS.logo}
          className="h-12 w-auto object-contain mx-auto drop-shadow-sm"
        />
        <h1 className="font-headline text-2xl sm:text-3xl font-extrabold text-[#343723]">
          <GsapTextHighlight>
            Join SideQuest
          </GsapTextHighlight>
        </h1>
        <p className="text-xs text-[#595a4a]">
          {role === 'traveller'
            ? 'Discover secret corridors, travel with compatible solo explorers and find local stays. Set up your travel profile next.'
            : 'Share a local stay, experience or place worth finding.'}
        </p>

        {/* Role Switcher */}
        <div className="inline-flex bg-[#f5f2e7] p-1 rounded-full border border-[#dedac8] mt-2">
          <button
            type="button"
            onClick={() => setRole('traveller')}
            className={`px-4 py-1.5 rounded-full text-xs font-bold transition-all ${
              role === 'traveller'
                ? 'bg-[#343723] text-[#EFE9D7] shadow-xs'
                : 'text-[#595a4a] hover:text-[#343723]'
            }`}
          >
            Traveller Account
          </button>
          <button
            type="button"
            onClick={() => setRole('collaborator')}
            className={`px-4 py-1.5 rounded-full text-xs font-bold transition-all ${
              role === 'collaborator'
                ? 'bg-[#7d2826] text-[#EFE9D7] shadow-xs'
                : 'text-[#595a4a] hover:text-[#343723]'
            }`}
          >
            Host / Collaborator
          </button>
        </div>
      </div>

      {/* Form */}
      <form
        onSubmit={handleRegister}
        className="bg-[#EFE9D7] p-5 sm:p-6 rounded-3xl shadow-sm border border-[#e8e4d4] space-y-4"
      >
        {role === 'traveller' ? (
          <>
            <div>
              <label className="text-xs font-bold text-[#343723]">Full Legal Name (as on Govt ID)</label>
              <input
                type="text"
                required
                placeholder="Your full name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="mt-1 w-full bg-[#f5f2e7] rounded-xl px-3.5 py-2.5 text-xs sm:text-sm text-[#343723] border border-[#dedac8] focus:outline-none focus:ring-2 focus:ring-[#343723]"
              />
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div>
                <label className="text-xs font-bold text-[#343723]">Email Address</label>
                <input
                  type="email"
                  required
                placeholder="you@example.com"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="mt-1 w-full bg-[#f5f2e7] rounded-xl px-3.5 py-2.5 text-xs sm:text-sm text-[#343723] border border-[#dedac8] focus:outline-none focus:ring-2 focus:ring-[#343723]"
                />
              </div>
              <div>
                <label className="text-xs font-bold text-[#343723]">Phone (+91)</label>
                <input
                  type="tel"
                  required
                  placeholder="98765 43210"
                  value={phone}
                  onChange={(e) => setPhone(e.target.value)}
                  className="mt-1 w-full bg-[#f5f2e7] rounded-xl px-3.5 py-2.5 text-xs sm:text-sm text-[#343723] border border-[#dedac8] focus:outline-none focus:ring-2 focus:ring-[#343723]"
                />
              </div>
            </div>
            <div>
              <label className="text-xs font-bold text-[#343723]">Password</label>
              <input type="password" required minLength={8} value={password} onChange={e => setPassword(e.target.value)} placeholder="At least 8 characters" className="mt-1 w-full bg-[#f5f2e7] rounded-xl px-3.5 py-2.5 text-sm border border-[#dedac8] focus:outline-none focus:ring-2 focus:ring-[#343723]" />
            </div>
          </>
        ) : (
          <>
            <div><label className="text-xs font-bold text-[#343723]">Your name</label><input required value={name} onChange={e => setName(e.target.value)} placeholder="Contact person" className="mt-1 w-full bg-[#f5f2e7] rounded-xl px-3.5 py-2.5 text-sm border border-[#dedac8]" /></div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div><label className="text-xs font-bold text-[#343723]">Email address</label><input type="email" required value={email} onChange={e => setEmail(e.target.value)} placeholder="you@example.com" className="mt-1 w-full bg-[#f5f2e7] rounded-xl px-3.5 py-2.5 text-sm border border-[#dedac8]" /></div>
              <div><label className="text-xs font-bold text-[#343723]">Password</label><input type="password" required minLength={8} value={password} onChange={e => setPassword(e.target.value)} placeholder="At least 8 characters" className="mt-1 w-full bg-[#f5f2e7] rounded-xl px-3.5 py-2.5 text-sm border border-[#dedac8]" /></div>
            </div>
            <div>
              <label className="text-xs font-bold text-[#343723]">Business or place name</label>
              <input
                type="text"
                required
                placeholder="Name visitors know"
                value={hostName}
                onChange={(e) => setHostName(e.target.value)}
                className="mt-1 w-full bg-[#f5f2e7] rounded-xl px-3.5 py-2.5 text-xs sm:text-sm text-[#343723] border border-[#dedac8] focus:outline-none focus:ring-2 focus:ring-[#7d2826]"
              />
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div><label className="text-xs font-bold text-[#343723]">Type of business</label><select required value={businessType} onChange={e=>setBusinessType(e.target.value)} className="mt-1 w-full rounded-xl border border-[#dedac8] bg-[#f5f2e7] px-3.5 py-3 text-sm"><option value="">Choose a type</option><option value="place">Place or attraction</option><option value="food">Food or drink</option><option value="guide">Local guide</option><option value="stay">Accommodation</option><option value="activity">Activity or experience</option></select></div>
              <div>
                <label className="text-xs font-bold text-[#343723]">Host Contact Phone</label>
                <input
                  type="tel"
                  required
                  placeholder="+91 91234 56789"
                  value={phone}
                  onChange={e => setPhone(e.target.value)}
                  className="mt-1 w-full bg-[#f5f2e7] rounded-xl px-3.5 py-2.5 text-xs sm:text-sm text-[#343723] border border-[#dedac8] focus:outline-none focus:ring-2 focus:ring-[#7d2826]"
                />
              </div>
            </div>

            <div className="p-3.5 bg-[#e5c9be]/40 rounded-xl border border-[#c98d7f] text-xs text-[#62201f] space-y-1.5">
              <span className="font-bold block flex items-center gap-1.5">
                <span className="material-symbols-outlined text-[16px]">verified_user</span>
                Collaborator Verification &amp; AI Hidden Gem Listing:
              </span>
              <p>Upload one government ID and real location-stamped place photos in your host workspace.</p>
              <p>SideQuest’s model reviews your listing. Cases without enough evidence go to an administrator; approval is not guaranteed.</p>
            </div>
          </>
        )}

        {/* Safety and Terms Checkbox */}
        <div className="pt-2">
          <label className="flex items-start gap-2.5 cursor-pointer text-xs text-[#595a4a]">
            <input
              type="checkbox"
              required
              checked={agreedToTerms}
              onChange={(e) => setAgreedToTerms(e.target.checked)}
              className="mt-0.5 w-4 h-4 rounded text-[#343723] focus:ring-[#343723]"
            />
            <span>
              I agree to verify my identity via government credentials and comply with{' '}
              <button
                type="button"
                onClick={() => onNavigate('terms')}
                className="font-bold text-[#343723] hover:underline"
              >
                Community Safety Guidelines &amp; Terms
              </button>
              .
            </span>
          </label>
        </div>

        <button
          type="submit"
          className={`w-full py-3.5 text-[#EFE9D7] text-xs sm:text-sm font-bold rounded-xl shadow-md transition-all active:scale-98 ${
            role === 'traveller' ? 'bg-[#343723] hover:bg-[#51533c]' : 'bg-[#7d2826] hover:bg-[#62201f]'
          }`}
        >
          {isSubmitting ? 'Creating account…' : role === 'traveller' ? 'Join the Expedition' : 'Submit Host Application'}
        </button>
      </form>

      <div className="text-center">
        <p className="text-xs text-[#595a4a]">
          Already have an account?{' '}
          <button
            type="button"
            onClick={() => onNavigate('login')}
            className="font-bold text-[#343723] hover:underline"
          >
            Sign In Here
          </button>
        </p>
      </div>
    </div>
  );
};
