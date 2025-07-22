// Adapted from the BFH year book idea at https://www.bfh.ch/dam/jcr:e512ae31-a3ed-4b65-b589-870383d794b0/abschlussarbeiten-bsc-informatik.pdf

#import "@preview/isc-hei-exec-summary:0.5.3": *

#let doc_language = "en" // Valid values are en, fr

// Must be < 425 characters long.
#let summary = "
PETAL is a multi-agent AI system that supports municipalities in energy planning by coordinating specialized agents through an orchestration layer.
By utilizing publicly available geospatial data and regulatory frameworks, the solution produces data-driven, context-aware recommendations that comply with sustainability objectives."

#let content = [

  // This is where you put the content of your executive summary
  == Objectives
  The primary objective of PETAL is to support municipalities in Valais/Wallis in the complex task of urban energy planning by providing an AI-powered, multi-agent system that processes various data sources to generate actionable insights. By combining geospatial data with legal and design documents, PETAL helps municipalities align with federal and cantonal sustainability goals.

  The system uses an orchestration layer to manage specialized AI agents, guiding their workflow based on the ongoing conversational context.
  PETAL delivers its recommendations through a web interface that visualizes data layers on a map, facilitating the interpretation for decision-makers.
  Ultimately, the project aims to support decision-making and promote energy strategies tailored to local contexts and energy landscapes.


  == Explanation
  The development of PETAL followed a research-driven, iterative methodology, combining AI system design with practical experimentation.
  The project started with an analysis of the main steps involved in urban energy planning, along with an assessment of relevant data sources and regulatory frameworks.
  AI agents were subsequently designed and developed for specific roles using targeted prompts, enabling tasks such as identifying the underlying intent of queries, interpreting information, and profiling energy usage.

  Evaluation was then conducted through both expert assessment and an automated benchmarking framework using language models to assess performance. Iterative improvements were made based on continuous feedback and observed limitations. Emphasis was placed on transparency, reproducibility, and adaptability to ensure the system can support future extensions and integration with existing municipal workflows.

  #colbreak() // As Typst does not support auto column balancing, this must be put to break the columns evenly. Move it to a location that makes the columns even.

  == Conclusion / Benefits
  This work demonstrates that AI-powered multi-agent systems can effectively support municipal energy planning by structuring this complex task into specialized, coordinated subtasks, each assigned to specific agents.
  The evaluation of PETAL shows strong capabilities in contextual reasoning and generation of data-grounded recommendations, with particularly good results observed when leveraging larger language models.
  However, challenges such as occasional inconsistencies and unsupported claims highlight the importance of continued efforts to enhance the system’s reliability and trustworthiness.
  PETAL lays the groundwork for advancing research and practical implementation of AI solutions within the field of energy planning.
  // Optionally, if you need a figure spanning multiple columns, you can use this.
  #place(
    bottom,
    scope: "parent",
    float: true,
    figure(
      image("figs/petal_interface_water.png", fit: "contain", height: 7cm, width: 100%),
      caption: "PETAL web interface displaying a follow-up report on small hydropower potential in Sion. The interface includes a chat panel showing the system response and an interactive map, visualizing water bodies and rooftop-level solar energy.",
    ),
  )
  // This is the end !
]

// TODO: please modify the following to suit your needs.
#show: project.with(
  title: "Optimisation de la planification énergétique\nurbaine par l'orchestration de l'IA",
  language: doc_language, // Modify global if required, see above
  authors: "Dion Osmani",
  student-picture: image("figs/portrait.jpg"), // [Optional], put none if not used
  permanent-email: none, // [Optional], put none if not used
  video-url: none, // This is a link to the video of you project, if any

  summary: summary, // Not to be changed
  content: content, // Not to be changed

  thesis-supervisor: "Jessen Page",
  thesis-co-supervisor: "Florian Desmons", // Optional, use none if not needed
  thesis-expert: "Nils Schüler", // Optional, use none if not needed
  academic-year: "2024-2025", // Optional, use none if not needed

  is-executive-summary: true, // This is an executive summary, not a full thesis

  school: "Haute École d'Ingénierie de Sion",
  programme: "Informatique et Systèmes de communication (ISC)",

  // Some keywords related to your thesis
  keywords: ("engineering", "data", "large language models", "AI agents", "energy planning"),
  major: "Data engineering", // "Software engineering", "Embedded systems", "Security", "something else"

  bind: right, // Bind the left side of the page
  footer: "Executive summary", // align(right, text(0.9em)[This is some content for the footer])
)
