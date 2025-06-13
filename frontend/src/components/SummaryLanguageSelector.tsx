import React from 'react';
import { useTranslation } from 'react-i18next';
import './SummaryLanguageSelector.css';

interface SummaryLanguageSelectorProps {
  currentLanguage: string;
  onLanguageChange: (language: string) => void;
  isLoading: boolean;
}

const SummaryLanguageSelector: React.FC<SummaryLanguageSelectorProps> = ({ 
  currentLanguage, 
  onLanguageChange,
  isLoading
}) => {
  const { t } = useTranslation('common');
  
  // Debug log to check the current language
  console.log("Langue du résumé actuelle:", currentLanguage);
  
  // Ensure the current language is one of the valid values
  const normalizedLanguage = currentLanguage === 'en' ? 'en' : 'fr';
  
  return (
    <div className="summary-language-selector">
      <div className="selector-label">{t('results.summaryLanguage')}:</div>
      <div className="selector-buttons">
        <button 
          className={`language-btn ${normalizedLanguage === 'fr' ? 'active' : ''}`}
          onClick={() => onLanguageChange('fr')}
          disabled={isLoading || normalizedLanguage === 'fr'}
        >
          {t('language.fr')}
        </button>
        <button 
          className={`language-btn ${normalizedLanguage === 'en' ? 'active' : ''}`}
          onClick={() => onLanguageChange('en')}
          disabled={isLoading || normalizedLanguage === 'en'}
        >
          {t('language.en')}
        </button>
      </div>
    </div>
  );
};

export default SummaryLanguageSelector;