#import "@preview/isc-hei-bthesis:0.5.0": *

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
#include "pages/abstract.typ"

#cleardoublepage()
#include "pages/résumé.typ"

#cleardoublepage()
#include "pages/acknowledgements.typ"

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
Data is emitted by various sources ; sensors, energy models or citizen records for e.g. all yield datapoints that help us assess different indicators we are willing to measure against our municipality. These indicators, _in fine_, help us evaluate our progress towards that goal.

Over the past two years, artificial intelligence (AI) has rapidly transformed our habits when interacting with information.

Large language models (LLMs) allow users to interact with these systems in natural language facilitating the interface between humans and _machines_. They can provide insights into vast amounts of data at speed and scales which are beyond our capabilities.

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

#highlight[TODO: provide a brief overview of the structure of the thesis (plan), add reference to extra scope in methodology?]

= Methodology <methodology>
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
