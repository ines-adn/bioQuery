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
  
  // Log de débogage pour vérifier la langue actuelle
  console.log("Langue du résumé actuelle:", currentLanguage);
  
  // Assurez-vous que la langue actuelle est l'une des valeurs valides
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