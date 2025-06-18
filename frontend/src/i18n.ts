import i18n from "i18next";
import { initReactI18next } from "react-i18next";

export const SUPPORTED_LANGUAGES = ["en", "fr", "de"];

function detectLanguage() {
  // get user prefered language
  const stored = localStorage.getItem("language");
  if (stored && SUPPORTED_LANGUAGES.includes(stored)) return stored;

  const browserLang =
    (navigator.languages && navigator.languages[0]) ||
    navigator.language ||
    "en";
  const shortLang = browserLang.split("-")[0];
  if (SUPPORTED_LANGUAGES.includes(shortLang)) return shortLang;

  return "en";
}

const defaultLanguage = detectLanguage();

i18n.use(initReactI18next).init({
  resources: {
    en: {
      translation: {
        side_panel_close_label: "Close",
        side_panel_new_conversation_label: "New conversation",
        side_panel_empty_label: "No conversations yet",
        side_panel_today_label: "Today",
        side_panel_last_week_label: "Last week",
        side_panel_older_label: "Previous",
        welcome_message: "Hey, what can I help with ?",
        prompt_propositions: [
          "Analyze the energy consumption trends of the past 5 years.",
          "Identify underutilized data sources that could improve energy demand forecasting.",
          "Assess the risks of grid overload during peak winter demand.",
          "Simulate the impact of a 20% increase in electric vehicle adoption on grid demand.",
        ],
        prompt_placeholder: "Type your message here...",
        indicator_message_great: "Keep being green!",
        indicator_message_ok: "Try a bit more.",
        indicator_message_bad: "You could make an effort...",
      },
    },
    fr: {
      translation: {
        side_panel_close_label: "Fermer",
        side_panel_new_conversation_label: "Nouvelle conversation",
        side_panel_empty_label: "Aucune conversation pour le moment",
        side_panel_today_label: "Aujourd'hui",
        side_panel_last_week_label: "La semaine dernière",
        side_panel_older_label: "Précédent",
        prompt_placeholder: "Tapez votre message ici...",
        welcome_message: "Comment puis-je vous aider ?",
        prompt_propositions: [
          "Analyse les tendances de la consommation d'énergie des 5 dernières années.",
          "Identifie les sources d'énergies sous-utilisées qui pourraient améliorer notre prévision.",
          "Évalue les risques de surcharge du réseau pendant la demande hivernale de pointe.",
          "Quel est l'impact d'une augmentation de 20 % de l'adoption des véhicules électriques.",
        ],
        indicator_message_great: "Continuez à être vert !",
        indicator_message_ok: "Essayez un peu plus.",
        indicator_message_bad: "Vous pourriez faire un effort...",
      },
    },
    de: {
      translation: {
        side_panel_close_label: "Schließen",
        side_panel_new_conversation_label: "Neues Gespräch",
        side_panel_empty_label: "Noch keine Unterhaltungen",
        side_panel_today_label: "Heute",
        side_panel_last_week_label: "Letzte Woche",
        side_panel_older_label: "Früher",
        prompt_placeholder: "Geben Sie hier Ihre Nachricht ein...",
        welcome_message: "Hallo, womit kann ich helfen?",
        prompt_propositions: [
          "Analysieren Sie die Energieverbrauchstrends der letzten 5 Jahre.",
          "Ungenutzte Datenquellen zur besseren Prognose identifizieren.",
          "Bewerten Sie die Risiken einer Netzüberlastung während der Spitzenlast im Winter.",
          "Simulieren Sie die Auswirkungen von 20 % mehr E-Autos auf das Netz.",
        ],
        indicator_message_great: "Bleiben Sie weiterhin grün!",
        indicator_message_ok: "Versuchen Sie es ein wenig mehr.",
        indicator_message_bad: "Sie könnten sich mehr anstrengen...",
      },
    },
  },
  lng: defaultLanguage,
  fallbackLng: "en",
  interpolation: {
    escapeValue: false,
  },
});

// listen for language changes and store in localStorage
i18n.on("languageChanged", (lng) => {
  localStorage.setItem("language", lng);
});

export default i18n;
