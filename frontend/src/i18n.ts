import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import LanguageDetector from 'i18next-browser-languagedetector';

// Import des fichiers de traduction
import translationEN from './locales/en/common.json';
import translationFR from './locales/fr/common.json';

const resources = {
  en: {
    common: translationEN
  },
  fr: {
    common: translationFR
  }
};

i18n
  // Détecte la langue de l'utilisateur (navigateur)
  .use(LanguageDetector)
  // Passe l'instance i18n à react-i18next
  .use(initReactI18next)
  // Initialisation de i18next
  .init({
    resources,
    fallbackLng: 'fr', // Langue par défaut si la langue détectée n'est pas disponible
    defaultNS: 'common', // Namespace par défaut
    
    interpolation: {
      escapeValue: false, // React fait déjà l'échappement
    },
    
    detection: {
      order: ['querystring', 'cookie', 'localStorage', 'navigator'],
      lookupQuerystring: 'lang', // paramètre d'URL pour changer de langue (?lang=en)
      lookupCookie: 'i18next',
      lookupLocalStorage: 'i18nextLng',
      caches: ['localStorage', 'cookie'],
    }
  });

export default i18n;