#import "@preview/isc-hei-bthesis:0.5.2": *

#let doc_language = "en" // Valid values are en, fr

#show: project.with(
  title: "Optimisation de la planification énergétique urbaine par l'orchestration de l'IA",
  sub-title: "A Bachelor Thesis in Data Engineering", // Optional

  is-thesis: true,
  split-chapters: true,

  thesis-supervisor: "Prof. Jessen Page",
  thesis-co-supervisor: "Florian Desmons", // Optional, use none if not needed
  thesis-expert: "Nils Schüler", // Optional, use none if not needed

  school: "Haute École d'Ingénierie de Sion",
  programme: "Informatique et Systèmes de communication (ISC)",

  // Some keywords related to your thesis
  keywords: ("engineering", "data", "large language models", "AI agents", "energy planning"),
  major: "Data engineering", // "Software engineering", "Embedded systems", "Security", "something else"

  authors: "Dion Osmani",

  date: datetime(year: 2025, month: 6, day: 24), // or datetime.today()
  language: doc_language, // en or fr
  version: none, // or for instance "1.0", for the version of your thesis],
  code-theme: "bluloco-light",
)

// // If using acronyms
#import "@preview/acrostiche:0.5.2": *
#include "acronyms.typ"

// Let's get started folks!

#cleardoublepage()
#heavy-title("Abstract")

The abstract of a bachelor thesis should provide a concise summary of the entire work. It typically includes:

- The context and motivation for the research.
- The main objective or research question.
- A brief description of the methodology or approach used.
- The key results or findings.
- The main conclusion or implications of the work.

The abstract should be self-contained, clear, and usually does not exceed 250–300 words. It allows readers to quickly understand the purpose and outcomes of the thesis without reading the full document.

The abstract *must* be written in both French and English.

Please also insert your project git/github URL HERE if your project is not confidential.

#lorem(150)

#v(1fr)

*Keywords* : #context inc.global-keywords.get().join(", ")

#cleardoublepage()
#heavy-title("Résumé")

Le résumé d’un mémoire de bachelor doit fournir un aperçu concis de l’ensemble du travail. Il inclut généralement :

- Le contexte et la motivation de la recherche.
- L’objectif principal ou la question de recherche.
- Une brève description de la méthodologie ou de l’approche utilisée.
- Les principaux résultats ou découvertes.
- La conclusion principale ou les implications du travail.

Le résumé doit être autonome, clair et ne pas dépasser habituellement 250 à 300 mots. Il permet aux lecteurs de comprendre rapidement le but et les résultats du mémoire sans lire l’intégralité du document.

Le résumé doit être rédigé en français *et* en anglais.

Veuillez également ajouter l'URL de votre git/github ici si le projet n'est pas confidential.

#lorem(150)

#v(1fr)

*Keywords* : #context inc.global-keywords.get().join(", ")

#cleardoublepage()
// Get the proper title for acknowledgements if not written in English
#heavy-title(context i18n(inc.global-language.get(), "acknowledgements"))

The *Acknowledgements* section of a bachelor thesis is where you express gratitude to those who supported you during your research and writing process. It is an *OPTIONAL* section. It may include:

- Academic supervisors or advisors who provided guidance.
- Professors or instructors who offered feedback or resources.
- Family and friends for emotional or practical support.
- Institutions or organizations that provided funding, facilities, or data.
- Anyone else who contributed significantly to your work.

Keep this section concise and sincere. It is typically placed after the abstract and before the main content of your thesis.


#table-of-contents(depth: 2)

= Writing a thesis

// Enable headers and footers from this point on
#set-header-footer(true)

Writing a report is an exercise that involves both *content and form*. In this document, we aim to simplify the formatting aspect without making any assumptions about the content, specifically in the context of the ISC program#footnote[Here is how to add a footnote https://isc.hevs.ch].

== The content of a thesis

The general structure of a bachelor thesis typically includes the following sections:

1. *Abstract*: A concise summary of the thesis, including the research question, methodology, results, and conclusions.
2. *Résumé*: A summary of the thesis in French.
3. *Acknowledgements*: (Optional) A section to thank those who supported the work.
4. *Table of Contents*: An organized listing of chapters and sections.
5. *Introduction*: Presents the background, motivation, objectives, and scope and plan of the thesis.
6. *State of the Art / Literature Review*: Reviews existing research and situates the thesis within the academic context. If salient in your work.
7. *Methodology*: Describes the methods, materials, and procedures used in the research / thesis.
8. *Results*: Presents the findings of the research, often with tables, figures, and analysis.
9. *Discussion*: Interprets the results, discusses implications, and relates findings to the research question.
10. *Conclusion*: Summarizes the main findings, contributions, and suggests future work.
11. *References / Bibliography*: Lists all sources cited in the thesis.
12. *Appendices*: (Optional) Contains supplementary material such as raw data, code, or additional explanations.

This structure may vary depending on the field of study, but these elements are commonly found in most bachelor theses. They are compulsory for the _ISC Bachelor thesis_.

= Introduction

Over the past few decades, society has been sensitized and slowly became more aware of significant problems that we are likely to face in the coming years.

Climate change and other environmental issues arise as a result of human-driven activities.

Scientists have monitored this matter and proposed various frameworks to address and mitigate these problems. In Switzerland, these different frameworks are implemented in the legislation and guidelines (at federal and canton levels) to steer the country towards a more sustainable future.

Municipalities in Switzerland are required to submit an energy planning document which outlines their future strategies to comply with those directives while also considering the characteristics of their energetical landscape.

These different properties can be quantified and analyzed through the use of a very valuable resource: data.
Data is emitted by various sources ; sensors, energy models or citizen records for e.g. all yield datapoints that help us assess different indicators we are willing to measure against our municipality. These indicators, _in fine_, help us evaluate our progress towards energy-related goals.

Over the past two years, artificial intelligence (AI) has rapidly transformed our habits when interacting with information.

Large Language Models (LLMs) allow users to interact with these systems in natural language facilitating the interface between humans and _machines_. They can provide insights into vast amounts of data at speed and scales which are beyond our capabilities.

This work tackles this exact problem that is the implementation of a solution which assists users into energy planning for a municipality.

This complex problem is approached by leveraging the power of _specialized_ AIs which each offer expertise into a variety of domains and are coordinated using an _orchestration_ AI to provide a solution. The expertise of the user interacting with the system tailors this solution to the specificities of the municipality.

The key steps in the engineering of this implementation are identifying the key information and datasources that are relevant to this process, structure the different AIs into an architecture whose components and interfaces are well-defined and ultimately implementing the solution.

The main objective of this work is to investigate how effective and reliable such a solution is and to assess its strengths and weaknesses. Additional goals are also outlined:
- Defining the important information in assisting user decision making.
- Understanding which datasources are available and relevant for this decision making.
- Structuring the decision by using an orchestration AI and many specialized AIs.
- Training specialized AIs on specialized datasets.
- Simplifying the user interface by handling the communication between the orchestration AI and the specialized AIs.

The project is scoped to municipalities within the canton of Valais/Wallis and strictly relies on publicly available data.
Certain measures are taken to ensure the privacy and security of data that would not be of public order considering the future implementation of extra datasources.

The solution is designed to offer a user-friendly interface from which users can interact with the system in a conversational manner and visualize a map of the municipality with different layers.

User behaviour is analyzed to takeaway user preferences which gradually adapt the answers to better meet the user expectations.

#highlight(
  "TODO: provide a brief overview of the structure of the thesis (plan), add reference to extra scope in methodology?",
)

= State of the Art

Large language models are highly effective tools for natural language processing and offer various opportunities to enhance our day-to-day tasks and workflows.
Ever since they have been introduced to the public, they have been adopted across a wide range of fields and applications.

#highlight("TODO: citer proprement")

The AI Institude at ITMO University published in 2025 a paper titled _LLM Agents for Smart City Management: Enhancing Decision Support Through Multi-Agent AI Systems_. The study examines how the natural language processing strengths of LLMs, combined with the distributed problem-solving abilities of multi-agent systems, can enhance urban decision-making processes.

The research focused on the testing of three hypotheses: (1) evaluating the capability of LLM agents to effectively route and process diverse urban queries against existing urban information systems, (2) the effectiveness of Retrieval-Augmented Generation (RAG) technology in improving response accuracy when working with local knowledge and regulations and (3) the impact of integrating LLM agents with existing urban information-systems - increasing efficiency and decreasing the decision making process time.

Their proposed solution was tested against 150 question-answer pairs and used St. Petersburg's Digital Urban Platform as a testbed.
The testing dataset was curated and built by a group of human experts such as specialists in urban data analysis, GIS specialists, and urban architects.

They then evaluated different configurations of LLM agents and state-of-the-art models against two primary metrics: G-eval and Answer Relevancy (AR).
The G-eval metric provides greater compliance with human requirements as it uses LLMs to evaluate answers from other LLMs based on custom user criteria. These criteria can for e.g. be provided as a list of rules specifying precise steps the LLM should take for evaluation, mirroring human reasoning process.
The AR metric, on the other hand, assesses the relevance of the answer from the LLM when compared with the correct answer provided by the experts. This process also leverages LLMs as it first extracts the different statements from the answer and then compares those to the reference answer.

In summary, the results show greater performance when integrating the RAG technology and urban information-systems to the solution (G-eval scores of 0.68-0.74) compared to standalone LLM responses (0.30-0.38).
They also concluded that this research proved practical real-world city management application as it enables efficient processing of urban planning tasks while maintaining high relevance in responses and shortening task completion time from days to hours.

The ITMO study presents a research-driven implementation of LLM agents focusing on decision support through integration with urban data platforms which curate and process urban data to provide insights and recommendations for urban planning and management.
Rather than relying on these large-scale platforms, the present thesis explores the potential of leveraging publicly available data from federal and cantonal sources while also considering the interface with municipal archives and residents' files.
This work strongly values the user experience with efforts in enhancing conversational interactions through preference-driven reporting and improving the clarity and quality of reported decisions.
It neither serves as a continuation nor a re-implementation of the ITMO study, but rather represents an independent application of AI agents to a related use case specifically adapted to the context of a Bachelor's thesis and shaped by my practical implementation choices and problem-solving approach.
Any solution designed and implemented around similar goals and data-related constraints, regardless of the specificities of the use case, may result in an architecture that is somewhat similar.

As we forget about the use case and consider more generic research on the matter, we encounter  publications that rather focus on the optimization of various aspects of AI agents, such as their scalability, efficiency, and robustness. While this work tackles some of these challenges, it does not incorporate major research efforts into these areas.

#highlight("TODO: citer aino différement?")
Another AI-centric solution relevant to this use case is #link("https://www.aino.world/")[aino], described on their homepage as an _AI GIS Analyst for Urban planning teams_. Developed and marketed in the United States, it is a commercial business solution offering a platform where an AI analyzes sites and provides visual insights from simple questions. This solution inspired me into a few design and user experience improvements in my work as no implementation details are provided.

These two points of view position this work within the fast-changing field of AI-driven solutions for urban planning and beyond.

= Methodology <methodology>

#highlight("TODO: faire des sous chapitres")

#lorem(800)

#lorem(800)

= Results
#lorem(950)

= Discussion
#lorem(1000)

= Conclusion
#lorem(1256)

//#bibliography("bibliography.bib", full: true, style: "ieee", title)
#pagebreak()
#the-bibliography(bib-file: "bibliography.bib", full: true, style: "ieee")

//////////////
// Appendices
//////////////
#cleardoublepage()
#appendix-page()
#pagebreak()

// Table of acronyms, NOT COMPULSORY
#print-index(
  title: heavy-title(i18n(doc_language, "acronym-table-title"), mult: 1, top: 1em, bottom: 1em),
  sorted: "up",
  delimiter: " : ",
  row-gutter: 0.7em,
  outlined: false,
)

#pagebreak()

// Table of listings
#table-of-figures()

// Code inclusion
#pagebreak()
#code-samples()

#let code_sample = read("code/sample.scala")

#figure(
  code()[
    #raw(code_sample, lang: "scala")
  ],
  caption: "Code included from the file example.scala",
)

#figure(
  code()[
    #raw(read("code/sort.py"), lang: "python")
  ],
  caption: "Second code included from the file example.scala",
)

#figure(
  code()[
    #raw(code_sample, lang: "scala")
  ],
  caption: "Second code included from the file example.scala",
)


// This is the end, folks!
