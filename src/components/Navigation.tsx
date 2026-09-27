import React, { useEffect, useRef, useState } from 'react';
import { AppScreen } from '../types';
import { ASSETS } from '../data/mockData';
import { api, type Me } from '../services/backend';

interface NavigationProps {
  currentScreen: AppScreen;
  onNavigate: (screen: AppScreen) => void;
  isDrawerOpen: boolean;
  setIsDrawerOpen: (open: boolean) => void;
  unreadChatCount?: number;
  onOpenNotifications?: () => void;
  hasUnreadNotifications?: boolean;
  onShowToast?: (message: string, icon?: string) => void;
}

export const Navigation: React.FC<NavigationProps> = ({
  currentScreen,
  onNavigate,
  isDrawerOpen,
  setIsDrawerOpen,
  unreadChatCount = 0,
  onOpenNotifications,
  hasUnreadNotifications = false,
  onShowToast,
}) => {
  const [profile, setProfile] = useState<Me | null>(null);
  const [uploadingPhoto, setUploadingPhoto] = useState(false);
  const avatarInput = useRef<HTMLInputElement>(null);
  useEffect(() => {
    let live = true;
    api.me().then((me) => { if (live) setProfile(me); }).catch(() => { if (live) setProfile(null); });
    return () => { live = false; };
  }, [currentScreen]);

  const handleAvatarUpload = async (file?: File) => {
    if (!file) return;
    if (!file.type.startsWith('image/')) { onShowToast?.('Choose an image file.', 'error'); return; }
    setUploadingPhoto(true);
    try {
      const { avatar_url } = await api.uploadAvatar(file);
      setProfile((current) => current ? { ...current, avatar_url } : current);
      onShowToast?.('Profile photo updated.', 'check_circle');
    } catch (error) {
      onShowToast?.(error instanceof Error ? error.message : 'Could not update your photo.', 'error');
    } finally { setUploadingPhoto(false); }
  };

  const initials = profile?.name?.trim().split(/\s+/).slice(0, 2).map((part) => part[0]).join('').toUpperCase() || '';
  const getScreenTitle = (screen: AppScreen) => {
    switch (screen) {
      case 'home':
        return 'Home';
      case 'routes':
        return 'Routes';
      case 'what-if':
        return 'What if?';
      case 'gems':
        return 'Gems';
      case 'solo-match':
        return 'Solo Match';
      case 'chats':
        return 'Chats';
      case 'pricing':
        return 'Pricing';
      case 'login':
        return 'Sign In';
      case 'register':
        return 'Join SideQuest';
      case 'preferences':
        return 'Travel profile';
      case 'terms':
        return 'Terms & Conditions';
      case 'collaborator':
        return 'Collaborator Portal';
      case 'admin':
        return 'Admin review';
      default:
        return 'SideQuest';
    }
  };

  return (
    <>
      {/* Top Fixed Header */}
      <header className="fixed top-0 inset-x-0 z-40 bg-[#efe9d7]/85 backdrop-blur-xl shadow-[0_1px_8px_rgba(0,0,0,0.03)] border-b border-[#e8e4d4]">
        <div className="h-16 px-4 max-w-7xl mx-auto flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            {/* Hamburger Button */}
            <button
              aria-label="Open menu"
              type="button"
              onClick={() => setIsDrawerOpen(true)}
              className="w-12 h-12 flex items-center justify-center rounded-xl text-[#343723] hover:bg-[#e8e4d4] active:scale-95 transition-all"
            >
              <span className="material-symbols-outlined text-[24px]">menu</span>
            </button>

            {/* Logo and Brand */}
            <button
              type="button"
              onClick={() => onNavigate('home')}
              className="flex items-center gap-2 hover:opacity-90 transition-opacity"
            >
              <img
                alt="SideQuest Logo"
                className="h-8 w-auto object-contain drop-shadow-xs"
                src={ASSETS.logo}
              />
              <span className="font-headline font-bold text-xl text-[#343723] tracking-tight">
                SideQuest
              </span>
            </button>
          </div>

          <nav aria-label="Main navigation" className="hidden items-center gap-1 lg:flex">
            {([
              ['routes','Plan a route'],...(profile?.role==='user'?[['what-if','What if?']]:[]),['gems','Hidden gems'],['solo-match','Travel match'],['chats','Chats'],
              ...(profile?.role==='admin'?[['admin','Admin']]:[]),
            ] as [AppScreen,string][]).map(([screen,label])=><button key={screen} type="button" onClick={()=>onNavigate(screen)} className={`rounded-full px-3 py-2 text-sm font-semibold transition ${currentScreen===screen?'bg-[#343723] text-[#f3efdf]':'text-[#595a4a] hover:bg-[#e8e4d4] hover:text-[#343723]'}`}>{label}</button>)}
          </nav>

          <div className="flex items-center gap-3 sm:gap-4">
            <span className="text-xs font-semibold text-[#595a4a] hidden sm:inline px-2 py-0.5 rounded-full bg-[#e8e4d4]">
              {getScreenTitle(currentScreen)}
            </span>

            {/* Notifications Button */}
            <button
              aria-label="Notifications"
              type="button"
              onClick={onOpenNotifications}
              className="w-12 h-12 flex items-center justify-center rounded-full text-[#595a4a] hover:text-[#343723] hover:bg-[#e8e4d4] relative transition-colors active:scale-95"
            >
              <span className="material-symbols-outlined text-[22px]">notifications</span>
              {hasUnreadNotifications && (
                <span className="absolute top-2.5 right-2.5 w-2.5 h-2.5 rounded-full bg-[#9b453e] ring-2 ring-[#efe9d7] animate-pulse"></span>
              )}
            </button>

            {/* Profile Avatar */}
            <input ref={avatarInput} type="file" accept="image/jpeg,image/png,image/webp" className="hidden" onChange={(event) => { void handleAvatarUpload(event.currentTarget.files?.[0]); event.currentTarget.value = ''; }} />
            <button type="button" onClick={() => profile ? avatarInput.current?.click() : onNavigate('login')} title={profile ? 'Change profile photo' : 'Sign in'} aria-label={profile ? 'Change profile photo' : 'Sign in'} disabled={uploadingPhoto} className="flex h-12 items-center gap-2 rounded-full transition-opacity hover:opacity-85 disabled:opacity-60">
              {profile?.avatar_url ? <img alt={`${profile.name}'s profile`} className="h-10 w-10 rounded-full object-cover ring-2 ring-[#898861]" src={profile.avatar_url} /> : <span aria-hidden="true" className="grid h-10 w-10 place-items-center rounded-full bg-[#dedac8] font-semibold text-[#343723] ring-2 ring-[#898861]">{uploadingPhoto ? '…' : initials || <span className="material-symbols-outlined text-[21px]">person</span>}</span>}
              {profile?.name && <span className="hidden max-w-28 truncate text-sm font-semibold text-[#343723] md:inline">{profile.name}</span>}
            </button>
          </div>
        </div>
      </header>

      {/* Navigation Drawer Backdrop */}
      <div
        className={`fixed inset-0 z-50 bg-[#343723]/50 backdrop-blur-xs transition-opacity duration-300 ${
          isDrawerOpen ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'
        }`}
        onClick={() => setIsDrawerOpen(false)}
      />

      {/* Slide-in Navigation Drawer */}
      <aside
        className={`fixed top-0 bottom-0 left-0 z-50 w-72 max-w-[82vw] bg-[#efe9d7] shadow-[0_8px_30px_rgba(15,23,42,0.15)] flex flex-col transition-transform duration-300 ease-out ${
          isDrawerOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        <div className="p-4 flex items-center justify-between border-b border-[#e8e4d4]">
          <div className="flex items-center gap-2">
            <img alt="SideQuest Logo" className="h-7 w-auto object-contain" src={ASSETS.logo} />
            <span className="font-headline font-bold text-lg text-[#343723]">SideQuest</span>
          </div>
          <button
            aria-label="Close menu"
            type="button"
            onClick={() => setIsDrawerOpen(false)}
            className="w-10 h-10 flex items-center justify-center rounded-xl text-[#595a4a] hover:bg-[#f5f2e7] transition-colors"
          >
            <span className="material-symbols-outlined text-[22px]">close</span>
          </button>

        </div>

        <div className="flex-1 overflow-y-auto px-3 py-3 space-y-1">
          {profile?.role !== 'user' && <button
            type="button"
            onClick={() => {
              onNavigate('home');
              setIsDrawerOpen(false);
            }}
            className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl transition-colors text-left ${
              currentScreen === 'home'
                ? 'bg-[#343723] text-[#EFE9D7] font-semibold shadow-xs'
                : 'text-[#595a4a] hover:bg-[#e8e4d4] hover:text-[#343723]'
            }`}
          >
            <span className="material-symbols-outlined text-[22px]">explore</span>
            <span className="text-sm font-medium">Home</span>
          </button>}

          {profile?.role === 'admin' && <button type="button" onClick={()=>{onNavigate('admin');setIsDrawerOpen(false)}} className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl transition-colors text-left ${currentScreen==='admin'?'bg-[#343723] text-[#EFE9D7] font-semibold':'text-[#595a4a] hover:bg-[#e8e4d4]'}`}><span className="material-symbols-outlined text-[22px]">admin_panel_settings</span><span className="text-sm font-medium">Admin review desk</span></button>}

          <button
            type="button"
            onClick={() => {
              onNavigate('routes');
              setIsDrawerOpen(false);
            }}
            className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl transition-colors text-left ${
              currentScreen === 'routes'
                ? 'bg-[#343723] text-[#EFE9D7] font-semibold shadow-xs'
                : 'text-[#595a4a] hover:bg-[#e8e4d4] hover:text-[#343723]'
            }`}
          >
            <span className="material-symbols-outlined text-[22px]">alt_route</span>
            <span className="text-sm font-medium">AI Route Planner</span>
          </button>

          {profile?.role === 'user' && <button
            type="button"
            onClick={() => { onNavigate('what-if'); setIsDrawerOpen(false); }}
            className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl transition-colors text-left ${currentScreen === 'what-if' ? 'bg-[#343723] text-[#EFE9D7] font-semibold shadow-xs' : 'text-[#595a4a] hover:bg-[#e8e4d4] hover:text-[#343723]'}`}
          >
            <span className="material-symbols-outlined text-[22px]">conversion_path</span>
            <span className="text-sm font-medium">What if?</span>
          </button>}

          <button
            type="button"
            onClick={() => {
              onNavigate('gems');
              setIsDrawerOpen(false);
            }}
            className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl transition-colors text-left ${
              currentScreen === 'gems'
                ? 'bg-[#343723] text-[#EFE9D7] font-semibold shadow-xs'
                : 'text-[#595a4a] hover:bg-[#e8e4d4] hover:text-[#343723]'
            }`}
          >
            <span className="material-symbols-outlined text-[22px]">diamond</span>
            <span className="text-sm font-medium">Hidden Gems</span>
          </button>

          <button
            type="button"
            onClick={() => {
              onNavigate('solo-match');
              setIsDrawerOpen(false);
            }}
            className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl transition-colors text-left ${
              currentScreen === 'solo-match'
                ? 'bg-[#343723] text-[#EFE9D7] font-semibold shadow-xs'
                : 'text-[#595a4a] hover:bg-[#e8e4d4] hover:text-[#343723]'
            }`}
          >
            <span className="material-symbols-outlined text-[22px]">diversity_1</span>
            <span className="text-sm font-medium">Solo Travel Match</span>
          </button>

          <button
            type="button"
            onClick={() => {
              onNavigate('chats');
              setIsDrawerOpen(false);
            }}
            className={`w-full flex items-center justify-between px-3 py-2.5 rounded-xl transition-colors text-left ${
              currentScreen === 'chats'
                ? 'bg-[#343723] text-[#EFE9D7] font-semibold shadow-xs'
                : 'text-[#595a4a] hover:bg-[#e8e4d4] hover:text-[#343723]'
            }`}
          >
            <div className="flex items-center gap-3">
              <span className="material-symbols-outlined text-[22px]">chat_bubble</span>
              <span className="text-sm font-medium">Chats & Squads</span>
            </div>
            {unreadChatCount > 0 && (
              <span className="px-2 py-0.5 text-xs font-bold rounded-full bg-[#9b453e] text-[#EFE9D7]">
                {unreadChatCount}
              </span>
            )}
          </button>

          <div className="my-2 h-px bg-[#d8d3be]/60" />

          <button
            type="button"
            onClick={() => {
              onNavigate('terms');
              setIsDrawerOpen(false);
            }}
            className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl transition-colors text-left ${
              currentScreen === 'terms'
                ? 'bg-[#343723] text-[#EFE9D7] font-semibold shadow-xs'
                : 'text-[#595a4a] hover:bg-[#e8e4d4] hover:text-[#343723]'
            }`}
          >
            <span className="material-symbols-outlined text-[22px]">verified_user</span>
            <span className="text-sm font-medium">Terms & Conditions</span>
          </button>

          {profile?.role !== 'user' && <button
            type="button"
            onClick={() => {
              onNavigate('collaborator');
              setIsDrawerOpen(false);
            }}
            className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl transition-colors text-left ${
              currentScreen === 'collaborator'
                ? 'bg-[#343723] text-[#EFE9D7] font-semibold shadow-xs'
                : 'text-[#343723] bg-[#e8e4d4]/60 hover:bg-[#e8e4d4]'
            }`}
          >
            <span className="material-symbols-outlined text-[22px]">storefront</span>
            <div className="flex flex-col">
              <span className="text-sm font-bold">Collaborator &amp; Host Hub</span>
              <span className="text-[10px] text-[#595a4a]">Verify Proofs &amp; List Gems</span>
            </div>
          </button>}

          {!profile && <button
            type="button"
            onClick={() => {
              onNavigate('register');
              setIsDrawerOpen(false);
            }}
            className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl transition-colors text-left ${
              currentScreen === 'register'
                ? 'bg-[#7d2826] text-[#EFE9D7] font-semibold shadow-xs'
                : 'text-[#7d2826] hover:bg-[#e5c9be]/60'
            }`}
          >
            <span className="material-symbols-outlined text-[22px]">person_add</span>
            <span className="text-sm font-medium">Create Account</span>
          </button>}
        </div>

        {profile?.role !== 'user' && <div className="p-4 border-t border-[#e8e4d4] bg-[#f5f2e7]">
          <button
            type="button"
            onClick={() => {
              if(profile){api.logout();setProfile(null);onNavigate('home');onShowToast?.('You have signed out.','check_circle')}
              else onNavigate('login');
              setIsDrawerOpen(false);
            }}
            className="w-full flex items-center gap-3 px-3 py-2 rounded-xl text-[#7d2826] hover:bg-[#e5c9be]/70 transition-colors"
          >
            <span className="material-symbols-outlined text-[22px]">{profile?'logout':'login'}</span>
            <span className="text-sm font-semibold">{profile?'Sign out':'Sign in'}</span>
          </button>
        </div>}
      </aside>

    </>
  );
};
