export type AppScreen =
  | 'home'
  | 'routes'
  | 'what-if'
  | 'gems'
  | 'solo-match'
  | 'chats'
  | 'pricing'
  | 'login'
  | 'register'
  | 'preferences'
  | 'terms'
  | 'collaborator'
  | 'admin';

export interface ProofDocument {
  id: string;
  type: 'govt_id' | 'property_proof' | 'tourism_license' | 'geotagged_photos';
  title: string;
  description: string;
  required: boolean;
  fileName?: string;
  fileSize?: string;
  status: 'pending' | 'uploaded' | 'verified';
  uploadedAt?: string;
  previewUrl?: string;
}

export interface CollaboratorProfile {
  name: string;
  businessName: string;
  email: string;
  phone: string;
  corridor: string;
  isVerified: boolean;
  verificationTier: string;
  documents: ProofDocument[];
  activeListingsCount: number;
}

export interface GemEvaluationResult {
  isGem: boolean;
  gemScore: number;
  /** Rule-based safety index (0-100); null when there is no location or no data. */
  safetyScore: number | null;
  safetyLevel?: 'high' | 'moderate' | 'low' | 'unknown';
  safetyReasons?: string[];
  safetyNote?: string;
  offbeatRating: number;
  /** null = not enough data to say (never guessed). */
  womenSafe: boolean | null;
  verdictTitle: string;
  analysis: string;
  curatorBadge: string;
  recommendedTags: string[];
  improvementSuggestions: string[];
}

export interface HiddenGem {
  id: string;
  name: string;
  category: string;
  categoryIcon: string;
  categoryColor: 'primary' | 'secondary' | 'tertiary';
  description: string;
  image: string;
  distanceKm: number;
  gemScore: number;
  safetyScore: number | null;
  womenSafe?: boolean | null;
  costLabel: string;
  hikeDurationOrFeature: string;
  badgeLabel?: string;
  tags: string[];
  lat?: number;
  lng?: number;
  isCollaboratorListed?: boolean;
  curatorHost?: string;
  aiEvaluation?: GemEvaluationResult;
  nearestHospitalKm?: number;
  nearestPoliceKm?: number;
  googleMapsUri?: string;
  photoAttributions?: { name: string; url?: string }[];
  nearestPharmacyKm?: number;
  routeTag?: string; // e.g. 'bengaluru-goa', 'coorg-chikmagalur', 'all'
  // filled by the hidden-gem recommender (/api/ml/hidden-gems)
  kind?: 'eat' | 'visit';
  rating?: number | null;
  reviews?: number | null;
  city?: string;
  safetyClass?: 'safe' | 'caution' | 'unknown' | 'avoid';
  safetyReasons?: string[];
  fromStartKm?: number;
  detourMin?: number;
  distanceNote?: string;
  whyReasons?: string[];
  youtubeVerified?: number;
}


export interface Waypoint {
  id: string;
  label: string;
  location: string;
  isOrigin?: boolean;
}

export interface SoloPeer {
  id: string;
  name: string;
  age: number;
  location: string;
  image: string;
  verifiedBadge: string;
  matchScore: number;
  travelDates: string;
  overlapDaysText: string;
  sharedPassions: string[];
  statsText: string;
}

export interface ChatMessage {
  id: string;
  sender: 'user' | 'peer';
  text: string;
  time: string;
  read?: boolean;
  waypointCard?: {
    title: string;
    gemScore: number;
    kmMark: string;
    description: string;
    image: string;
    activeNearby: number;
  };
}
