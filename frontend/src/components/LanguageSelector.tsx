import React from 'react';
import { useTranslation } from 'react-i18next';
import './LanguageSelector.css'; // Utiliser le fichier CSS dédié au composant

const LanguageSelector: React.FC = () => {
  const { i18n, t } = useTranslation('common');
  
  const changeLanguage = (lng: string) => {
    i18n.changeLanguage(lng);
  };
  
  return (
    <div className="language-selector">
      <button 
        className={`language-btn ${i18n.language === 'fr' ? 'active' : ''}`}
        onClick={() => changeLanguage('fr')}
      >
        {t('language.fr')}
      </button>
      <span className="language-divider">|</span>
      <button 
        className={`language-btn ${i18n.language === 'en' ? 'active' : ''}`}
        onClick={() => changeLanguage('en')}
      >
        {t('language.en')}
      </button>
    </div>
  );
};

export default LanguageSelector;