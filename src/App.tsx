import { lazy, Suspense, useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import type { AppScreen } from './types';
import { Navigation } from './components/Navigation';
import { NotificationModal } from './components/NotificationModal';
const HomeScreen = lazy(() => import('./components/screens/HomeScreen').then(({ HomeScreen }) => ({ default: HomeScreen })));
const RoutesScreen = lazy(() => import('./components/screens/RoutesScreen').then(({ RoutesScreen }) => ({ default: RoutesScreen })));
const WhatIfScreen = lazy(() => import('./components/screens/WhatIfScreen').then(({ WhatIfScreen }) => ({ default: WhatIfScreen })));
const GemsScreen = lazy(() => import('./components/screens/GemsScreen').then(({ GemsScreen }) => ({ default: GemsScreen })));
const SoloMatchScreen = lazy(() => import('./components/screens/SoloMatchScreen').then(({ SoloMatchScreen }) => ({ default: SoloMatchScreen })));
const ChatsScreen = lazy(() => import('./components/screens/ChatsScreen').then(({ ChatsScreen }) => ({ default: ChatsScreen })));
const PricingScreen = lazy(() => import('./components/screens/PricingScreen').then(({ PricingScreen }) => ({ default: PricingScreen })));
const LoginScreen = lazy(() => import('./components/screens/LoginScreen').then(({ LoginScreen }) => ({ default: LoginScreen })));
const RegisterScreen = lazy(() => import('./components/screens/RegisterScreen').then(({ RegisterScreen }) => ({ default: RegisterScreen })));
const PreferencesScreen = lazy(() => import('./components/screens/PreferencesScreen').then(({ PreferencesScreen }) => ({ default: PreferencesScreen })));
const TermsScreen = lazy(() => import('./components/screens/TermsScreen').then(({ TermsScreen }) => ({ default: TermsScreen })));
const CollaboratorScreen = lazy(() => import('./components/screens/CollaboratorScreen').then(({ CollaboratorScreen }) => ({ default: CollaboratorScreen })));
const AdminScreen = lazy(() => import('./components/screens/AdminScreen').then(({ AdminScreen }) => ({ default: AdminScreen })));
import LandingPage from './components/LandingPage';

interface ToastState {
  id: number;
  message: string;
  icon?: string;
}

export default function App() {
  const [currentScreen, setCurrentScreen] = useState<AppScreen>('home');
  const [isLanding, setIsLanding] = useState(true);
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);
  const [isNotificationsOpen, setIsNotificationsOpen] = useState(false);
  const [unreadChatCount, setUnreadChatCount] = useState(0);
  const [hasUnreadNotifications, setHasUnreadNotifications] = useState(false);
  const [toasts, setToasts] = useState<ToastState[]>([]);
  const [chatInitialPeerId, setChatInitialPeerId] = useState<string | undefined>(undefined);
  const [isOnline, setIsOnline] = useState(navigator.onLine);

  useEffect(() => {
    const handleOnline = () => setIsOnline(true);
    const handleOffline = () => setIsOnline(false);
    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);
    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);



  const showToast = (message: string, icon: string = 'info') => {
    const id = Date.now();
    setToasts((prev) => [...prev, { id, message, icon }]);
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 3000);
  };

  const handleNavigate = (screen: AppScreen) => {
    setIsLanding(false);
    setCurrentScreen(screen);
    window.scrollTo({ top: 0, behavior: 'smooth' });
    if (screen === 'chats') {
      setUnreadChatCount(0);
    }
  };

  const handleOpenNotifications = () => {
    setIsNotificationsOpen(true);
    setHasUnreadNotifications(false);
  };

  const renderScreen = () => {
    switch (currentScreen) {
      case 'home':
        return (
          <HomeScreen
            onNavigate={handleNavigate}
            onOpenGemPreview={() => handleNavigate('gems')}
          />
        );
      case 'routes':
        return <RoutesScreen onNavigate={handleNavigate} onShowToast={showToast} />;
      case 'what-if':
        return <WhatIfScreen onNavigate={handleNavigate} onShowToast={showToast} />;
      case 'gems':
        return <GemsScreen onNavigate={handleNavigate} onShowToast={showToast} />;
      case 'solo-match':
        return (
          <SoloMatchScreen
            onNavigate={handleNavigate}
            onShowToast={showToast}
            onSelectPeerForChat={(peerId) => {
              setChatInitialPeerId(peerId);
              handleNavigate('chats');
            }}
          />
        );
      case 'chats':
        return (
          <ChatsScreen
            onNavigate={handleNavigate}
            onShowToast={showToast}
            initialPeerId={chatInitialPeerId}
          />
        );
      case 'pricing':
        return <PricingScreen onNavigate={handleNavigate} onShowToast={showToast} />;
      case 'login':
        return <LoginScreen onNavigate={handleNavigate} onShowToast={showToast} />;
      case 'register':
        return <RegisterScreen onNavigate={handleNavigate} onShowToast={showToast} />;
      case 'preferences':
        return <PreferencesScreen onNavigate={handleNavigate} onShowToast={showToast} />;
      case 'terms':
        return <TermsScreen onNavigate={handleNavigate} onShowToast={showToast} />;
      case 'collaborator':
        return <CollaboratorScreen onNavigate={handleNavigate} onShowToast={showToast} />;
      case 'admin':
        return <AdminScreen onNavigate={handleNavigate} onShowToast={showToast} />;
      default:
        return <HomeScreen onNavigate={handleNavigate} />;
    }
  };

  return (
    <div className="min-h-screen bg-[#efe9d7] text-[#343723] flex flex-col font-body selection:bg-[#898861] selection:text-[#efe9d7]">
      {/* Top Header & Drawers */}
      {!isLanding && currentScreen !== 'preferences' && (
        <Navigation
          currentScreen={currentScreen}
          onNavigate={handleNavigate}
          isDrawerOpen={isDrawerOpen}
          setIsDrawerOpen={setIsDrawerOpen}
          unreadChatCount={unreadChatCount}
          onOpenNotifications={handleOpenNotifications}
          hasUnreadNotifications={hasUnreadNotifications}
          onShowToast={showToast}
        />
      )}

      {/* Offline Status Banner */}
      <AnimatePresence>
        {!isLanding && !isOnline && (
          <motion.div
            initial={{ y: -40, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            exit={{ y: -40, opacity: 0 }}
            className="fixed top-16 left-0 right-0 z-40 flex items-center justify-center gap-2 bg-[#343723] text-[#EFE9D7] text-xs font-semibold py-2 px-4 shadow-lg"
          >
            <span className="material-symbols-outlined text-[16px] text-[#e5c9be]">wifi_off</span>
            <span>You're offline — viewing cached content</span>
            <span className="ml-2 px-2 py-0.5 bg-[#7d2826] rounded-full text-[10px] font-bold">OFFLINE</span>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Content Area */}

      {isLanding ? (
        <LandingPage onNavigate={handleNavigate} />
      ) : (
        <main className={`flex-1 w-full ${currentScreen === 'preferences' ? '' : 'pt-16'}`}>
          <Suspense fallback={<div className="min-h-[55vh] grid place-items-center text-xs uppercase tracking-[0.18em] text-[#595a4a]" role="status">Opening your next sidequest…</div>}>
            <AnimatePresence mode="wait">
              <motion.div
                key={currentScreen}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                transition={{ duration: 0.32, ease: [0.22, 1, 0.36, 1] }}
                className="w-full"
              >
                {renderScreen()}
              </motion.div>
            </AnimatePresence>
          </Suspense>
        </main>
      )}

      {/* Radar Notifications Modal */}
      {!isLanding && <NotificationModal
        isOpen={isNotificationsOpen}
        onClose={() => setIsNotificationsOpen(false)}
        onNavigateToChat={() => handleNavigate('chats')}
      />}

      {/* Interactive Toast Notifications Pill Stack */}
      {!isLanding && <div className="fixed top-20 right-4 z-50 flex flex-col gap-2 pointer-events-none max-w-xs">
        <AnimatePresence>
          {toasts.map((toast) => (
            <motion.div
              key={toast.id}
              initial={{ opacity: 0, x: 20, scale: 0.95 }}
              animate={{ opacity: 1, x: 0, scale: 1 }}
              exit={{ opacity: 0, x: 20, scale: 0.95 }}
              className="pointer-events-auto bg-[#343723]/95 text-[#EFE9D7] backdrop-blur-md px-3.5 py-2.5 rounded-2xl shadow-xl border border-[#EFE9D7]/10 flex items-center gap-2.5 text-xs font-semibold"
            >
              <span className="material-symbols-outlined text-[#898861] text-[18px] flex-shrink-0">
                {toast.icon || 'info'}
              </span>
              <span className="leading-snug">{toast.message}</span>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>}
    </div>
  );
}
