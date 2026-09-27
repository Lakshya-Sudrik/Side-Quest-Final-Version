import React, { useState } from 'react';
import { AppScreen } from '../../types';
import { ASSETS } from '../../data/mockData';
import { GsapTextHighlight } from '../GsapTextHighlight';
import { api } from '../../services/backend';

interface LoginScreenProps {
  onNavigate: (screen: AppScreen) => void;
  onShowToast: (msg: string, icon?: string) => void;
}

export const LoginScreen: React.FC<LoginScreenProps> = ({
  onNavigate,
  onShowToast,
}) => {
  const [role, setRole] = useState<'roamer' | 'collaborator'>('roamer');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [rememberMe, setRememberMe] = useState(true);

  const handleRoleChange = (newRole: 'roamer' | 'collaborator') => {
    setRole(newRole);
    if (newRole === 'collaborator') {
      setEmail('');
    } else {
      setEmail('');
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    try {
      const result = await api.login(email.trim(), password, role === 'collaborator' ? 'collaborator' : 'user', rememberMe);
      onShowToast(`Welcome back, ${result.user.name}.`, 'check_circle');
      onNavigate(result.user.role === 'admin' ? 'admin' : result.user.role === 'collaborator' ? 'collaborator' : result.user.preferences ? 'routes' : 'preferences');
    } catch (error) {
      onShowToast(error instanceof Error ? error.message : 'Unable to sign in. Please try again.', 'error');
    } finally { setIsSubmitting(false); }
  };

  return (
    <div className="flex flex-col w-full max-w-md mx-auto px-4 pt-4 pb-20 space-y-5">
      <div className="flex flex-col items-center text-center space-y-2 pt-2">
        <img
          alt="SideQuest Logo"
          src={ASSETS.logo}
          className="h-14 w-auto object-contain drop-shadow-md"
        />
        <h1 className="font-headline text-2xl sm:text-3xl font-extrabold text-[#343723]">
          <GsapTextHighlight>
            {role === 'collaborator' ? 'Partner & Host Login' : 'Welcome back, Roamer'}
          </GsapTextHighlight>
        </h1>
        <p className="text-xs text-[#595a4a]">
          {role === 'collaborator'
            ? 'Sign in to audit proof documents, track visitor stats, and evaluate new places with Gemini AI.'
            : 'Sign in to access your offline trail maps, solo matches, and field journals.'}
        </p>
      </div>

      {/* Account Type Toggle Switcher */}
      <div className="bg-[#e8e4d4] p-1 rounded-2xl flex items-center">
        <button
          type="button"
          onClick={() => handleRoleChange('roamer')}
          className={`flex-1 py-2 rounded-xl text-xs font-bold transition-all flex items-center justify-center gap-1.5 ${
            role === 'roamer'
              ? 'bg-[#EFE9D7] text-[#343723] shadow-xs'
              : 'text-[#595a4a] hover:text-[#343723]'
          }`}
        >
          <span className="material-symbols-outlined text-[16px]">explore</span>
          <span>Explorer / Roamer</span>
        </button>

        <button
          type="button"
          onClick={() => handleRoleChange('collaborator')}
          className={`flex-1 py-2 rounded-xl text-xs font-bold transition-all flex items-center justify-center gap-1.5 ${
            role === 'collaborator'
              ? 'bg-[#343723] text-[#EFE9D7] shadow-xs'
              : 'text-[#595a4a] hover:text-[#343723]'
          }`}
        >
          <span className="material-symbols-outlined text-[16px]">storefront</span>
          <span>Host / Collaborator</span>
        </button>
      </div>

      <form onSubmit={handleSubmit} className="bg-[#EFE9D7] p-5 sm:p-6 rounded-3xl shadow-sm border border-[#e8e4d4] space-y-4">
        <div>
          <label className="text-xs font-bold text-[#343723]">Email address</label>
          <div className="mt-1 flex items-center bg-[#f5f2e7] rounded-xl px-3.5 py-2.5 border border-[#dedac8] focus-within:border-[#343723]">
            <span className="material-symbols-outlined text-[#343723] text-[20px] mr-2">
              mail
            </span>
            <input
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@example.com"
              className="w-full bg-transparent text-xs sm:text-sm text-[#343723] focus:outline-none"
            />
          </div>
        </div>

        <div>
          <div className="flex items-center justify-between">
            <label className="text-xs font-bold text-[#343723]">Password</label>
          </div>
          <div className="mt-1 flex items-center bg-[#f5f2e7] rounded-xl px-3.5 py-2.5 border border-[#dedac8] focus-within:border-[#343723]">
            <span className="material-symbols-outlined text-[#343723] text-[20px] mr-2">
              key
            </span>
            <input
              type={showPassword ? 'text' : 'password'}
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Enter your password"
              className="w-full bg-transparent text-xs sm:text-sm text-[#343723] focus:outline-none"
            />
            <button
              type="button"
              onClick={() => setShowPassword(!showPassword)}
              className="text-[#898861] hover:text-[#343723]"
            >
              <span className="material-symbols-outlined text-[18px]">
                {showPassword ? 'visibility_off' : 'visibility'}
              </span>
            </button>
          </div>
        </div>

        <div className="flex items-center justify-between text-xs">
          <label className="flex items-center gap-2 cursor-pointer text-[#595a4a]">
            <input
              type="checkbox"
              checked={rememberMe}
              onChange={(e) => setRememberMe(e.target.checked)}
              className="w-4 h-4 rounded text-[#343723] focus:ring-[#343723]"
            />
            <span>Remember this device</span>
          </label>
        </div>

        <button
          type="submit"
          className="w-full py-3.5 bg-[#343723] hover:bg-[#51533c] text-[#EFE9D7] text-xs sm:text-sm font-bold rounded-xl shadow-md transition-all active:scale-98 flex items-center justify-center gap-2"
        >
          {isSubmitting ? 'Signing in…' : role === 'collaborator' ? (
            <>
              <span className="material-symbols-outlined text-[18px]">storefront</span>
              <span>Access Collaborator Portal</span>
            </>
          ) : (
            <span>Resume Journey</span>
          )}
        </button>

        <div className="relative flex py-2 items-center">
          <div className="flex-grow border-t border-[#e8e4d4]"></div>
          <span className="flex-shrink mx-3 text-[11px] text-[#898861] uppercase">
            or continue with
          </span>
          <div className="flex-grow border-t border-[#e8e4d4]"></div>
        </div>

        <div className="grid grid-cols-2 gap-2.5">
          <button
            type="button"
            onClick={() => {
              onShowToast('Google sign-in is not connected yet.', 'info');
            }}
            className="flex items-center justify-center gap-2 py-2.5 px-3 bg-[#f5f2e7] hover:bg-[#dedac8] border border-[#dedac8] rounded-xl text-xs font-bold text-[#343723] transition-colors"
          >
            <span>Google</span>
          </button>
          <button
            type="button"
            onClick={() => {
              onShowToast('Apple sign-in is not connected yet.', 'info');
            }}
            className="flex items-center justify-center gap-2 py-2.5 px-3 bg-[#f5f2e7] hover:bg-[#dedac8] border border-[#dedac8] rounded-xl text-xs font-bold text-[#343723] transition-colors"
          >
            <span>Apple</span>
          </button>
        </div>
      </form>

      <div className="text-center space-y-1">
        <p className="text-xs text-[#595a4a]">
          New to SideQuest?{' '}
          <button
            type="button"
            onClick={() => onNavigate('register')}
            className="font-bold text-[#343723] hover:underline"
          >
            Create an Account
          </button>
        </p>
        <button
          type="button"
          onClick={() => onNavigate('terms')}
          className="text-[11px] text-[#898861] hover:underline block mx-auto"
        >
          Review Terms &amp; Safety Protocol
        </button>
      </div>
    </div>
  );
};
