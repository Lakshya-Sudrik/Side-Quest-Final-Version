import { HiddenGem, CollaboratorProfile, ProofDocument, GemEvaluationResult } from '../types';
import { CURATED_GEMS, ASSETS } from './mockData';

const STORAGE_KEY_GEMS = 'sidequest_custom_gems_v1';
const STORAGE_KEY_COLLABORATOR = 'sidequest_collaborator_profile_v1';

export const INITIAL_COLLABORATOR_DOCS: ProofDocument[] = [
  {
    id: 'doc-govt-id',
    type: 'govt_id',
    title: 'Government Identity Proof',
    description: 'Aadhaar Card, Passport, or Voter ID of the primary property manager or authorized local curator.',
    required: true,
    fileName: 'Govt_ID_RohanDeshmukh_Aadhaar.pdf',
    fileSize: '1.8 MB',
    status: 'verified',
    uploadedAt: 'Sep 02, 2025',
    previewUrl: 'https://images.unsplash.com/photo-1557804506-669a67965ba0?auto=format&fit=crop&w=600&q=80',
  },
  {
    id: 'doc-property-proof',
    type: 'property_proof',
    title: 'Property Ownership / Registered Lease Deed',
    description: 'Land record 7/12 extract, title deed, or registered long-term lease authorizing ecotourism / homestay access.',
    required: true,
    fileName: 'Kaveri_Orchard_LeaseDeed_Reg2023.pdf',
    fileSize: '4.2 MB',
    status: 'verified',
    uploadedAt: 'Sep 03, 2025',
  },
  {
    id: 'doc-tourism-license',
    type: 'tourism_license',
    title: 'State Tourism / Homestay / FSSAI Operating Permit',
    description: 'Official registration certificate from Department of Tourism or local Gram Panchayat sanitization NOC.',
    required: true,
    fileName: 'Goa_Tourism_Dept_Homestay_Cert_B22.pdf',
    fileSize: '2.1 MB',
    status: 'verified',
    uploadedAt: 'Sep 04, 2025',
  },
  {
    id: 'doc-geotagged-photos',
    type: 'geotagged_photos',
    title: 'Geo-Tagged Property & Trail Photos',
    description: 'Unfiltered high-resolution photographs with embedded EXIF GPS coordinates showing trail entrance, safety signs, and natural sanctuary.',
    required: true,
    fileName: 'GPS_Photos_Trailhead_Sanctuary_12Imgs.zip',
    fileSize: '18.4 MB',
    status: 'verified',
    uploadedAt: 'Sep 05, 2025',
    previewUrl: ASSETS.kaveriBamboo,
  },
];

export const INITIAL_COLLABORATOR: CollaboratorProfile = {
  name: 'Rohan Deshmukh',
  businessName: 'Kaveri River Orchards & Bamboo Grove Sanctuary',
  email: 'rohan.deshmukh@kaverigrove.in',
  phone: '+91 98231 44582',
  corridor: 'Western Ghats Scenic Spur (Chorla - Kaveri Ridge)',
  isVerified: true,
  verificationTier: 'Level 2 Verified Local Host',
  documents: INITIAL_COLLABORATOR_DOCS,
  activeListingsCount: 1,
};

export function getStoredCollaboratorProfile(): CollaboratorProfile {
  try {
    const data = localStorage.getItem(STORAGE_KEY_COLLABORATOR);
    if (data) {
      return JSON.parse(data);
    }
  } catch (e) {
    // localStorage unavailable
  }
  return INITIAL_COLLABORATOR;
}

export function saveCollaboratorProfile(profile: CollaboratorProfile): void {
  try {
    localStorage.setItem(STORAGE_KEY_COLLABORATOR, JSON.stringify(profile));
  } catch (e) {
    // localStorage unavailable
  }
}

export function getAllGems(): HiddenGem[] {
  let customGems: HiddenGem[] = [];
  try {
    const data = localStorage.getItem(STORAGE_KEY_GEMS);
    if (data) {
      customGems = JSON.parse(data);
    }
  } catch (e) {
    // localStorage unavailable
  }
  // Combine custom gems (at top) with default curated gems
  return [...customGems, ...CURATED_GEMS];
}

export function addApprovedGem(
  gemData: Omit<HiddenGem, 'id'>,
  aiEvaluation?: GemEvaluationResult
): HiddenGem {
  const id = `collab-gem-${Date.now()}`;
  const newGem: HiddenGem = {
    ...gemData,
    id,
    isCollaboratorListed: true,
    aiEvaluation,
  };

  try {
    let customGems: HiddenGem[] = [];
    const data = localStorage.getItem(STORAGE_KEY_GEMS);
    if (data) {
      customGems = JSON.parse(data);
    }
    customGems = [newGem, ...customGems];
    localStorage.setItem(STORAGE_KEY_GEMS, JSON.stringify(customGems));
  } catch (e) {
    console.error('Failed to save to localStorage:', e);
  }

  return newGem;
}
