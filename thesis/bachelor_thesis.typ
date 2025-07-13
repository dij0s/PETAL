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

#highlight("TODO: diviser en sections? contexte, problématique...")

Over the past few decades, society has been sensitized and slowly became more aware of significant problems that we are likely to face in the coming years.

Climate change and other environmental issues arise as a result of human-driven activities.

Scientists have monitored this matter and proposed various frameworks to address and mitigate these problems. In Switzerland, these different frameworks are implemented in the legislation and guidelines (at federal and canton levels) to steer the country towards a more sustainable future.
Municipalities may introduce in their regulations energy requirements that are more constraining than those set by the cantonal law.
#highlight("TODO: citer loi teams jessen?")

Municipalities in Switzerland are required to submit an energy planning document which outlines their future strategies to comply with those directives while also considering the characteristics of their energetical landscape.

These different properties can be quantified and analyzed through the use of a very valuable resource: data.
Data is emitted by various sources ; sensors, energy models or citizen records for e.g. all yield data points that help us assess different indicators we are willing to measure against our municipality. These indicators, _in fine_, help us evaluate our progress towards energy-related goals.

Over the past few years, Artificial Intelligence (AI) has rapidly transformed our habits when interacting with information.

Large Language Models (LLM) allow users to interact with these systems in natural language facilitating the interface between humans and _machines_. They can provide insights into vast amounts of data at speed and scales which are beyond our capabilities.

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

LLM agents (sometimes called AI agents) are software systems that use AI to pursue goals and complete tasks on behalf of users. They show reasoning, planning, and memory and have a level of autonomy to make decisions, learn, and adapt as per #link("https://cloud.google.com/discover/what-are-ai-agents?hl=en")[cloud.google.com].
#highlight("mieux citer??")

Their proposed solution was tested against 150 question-answer pairs and used St. Petersburg's Digital Urban Platform as a testbed.
The testing dataset was curated and built by a group of human experts such as specialists in urban data analysis, GIS specialists, and urban architects.

They then evaluated different configurations of LLM agents and state-of-the-art models against two primary metrics: G-eval and answer relevance.
The G-eval metric provides greater compliance with human requirements as it uses LLMs to evaluate answers from other LLMs based on custom user criteria. These criteria can for e.g. be provided as a list of rules specifying precise steps the LLM should take for evaluation, mirroring human reasoning process.
The answer relevance metric, on the other hand, assesses the relevance of the answer from the LLM when compared with the correct answer provided by the experts. This process also leverages LLMs as it first extracts the different statements from the answer and then compares those to the reference answer.
Both metrics are bounded between 0 and 1, where a higher value indicates better performance.

In summary, the results show greater performance when integrating the RAG technology and urban information-systems to the solution (G-eval scores of 0.68-0.74) compared to standalone LLM responses (G-eval scores of 0.30-0.38).
Relevance scores, on the other hand, remain high whatever the configuration as they are inherently designed to produce semantically relevant text.
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

The following chapter provides an overview of the methodology that has been adopted in this work and describes the approach taken to design, implement and further evaluate the proposed solution.
A clear emphasis is put onto the key decisions that have shaped the solution throughout its development as this chapter aims to offer full transparency and reproducibility which enables readers to understand the rationale and structure behind it.

== Requirements

Identifying and describing the primary requirements lays the groundwork for the design of the solution.
The main requirements were established and further refined as the project progressed to meet the expectations and project goals throughout ongoing discussions with the supervisors.
Those are categorized into functional and non-functional requirements. Functional requirements focus on user requirements and product features whereas non-functional requirements focus on user expectations and product properties.

The first step is to understand the problem that is solved. Energy planning typically involves:
- Identifying available energy resources#footnote[The resources and needs are assessed within the geographical boundaries of the municipality, only.], infrastructure and untapped potential.
- Characterizing the needs.
- Assessing different measures and their impacts ; sobriety (reducing energy consumption), efficiency (more efficient technologies) and production of renewable energy sources.
#highlight("TODO: décrire dans ce chapitre les données et autres à disposition?")
Hence, assisting users in energy planning requires a solution that can gather relevant data sources, analyze and present the current energy landscape of the municipality and provide actionable recommendations tailored to the context of the municipality while ensuring compliance with the legislation and guidelines that apply to said municipality.

Besides that, it had been requested that the user interface has a map showcasing the assessed data points within the municipality as well as for the AI to be able to _remember_ the user's preferences and past interactions to have the answer better fit what the user expects.

These initial requirements established the basis for the project. As the solution was evaluated with supervisors on a weekly basis, additional requirements emerged gradually shaping the solution to enhance the overall solution and meet the needs of energy planning.

#highlight("TODO: revoir la dernière phrase, ajouter des autres requirements ?")

These requirements are summarized in the #ref(<requirements_table>) below:

#show table.cell.where(y: 0): strong
#show figure: set block(breakable: true)
#set table(stroke: (x, y) => if y == 0 {
  (bottom: 0.7pt + black)
})

#figure(
  table(
    columns: 3,
    table.header([Requirement], [Type], [Description]),
    [Gather relevant data sources],
    [Functional],
    [The solution must collect and integrate data from various sources relevant to the municipality's energy landscape.],

    [Analyze and present energy landscape],
    [Functional],
    [The system should process and visualize the current energy situation of the municipality, enabling users to understand available resources and needs.],

    [Provide actionable recommendations],
    [Functional],
    [The solution must generate tailored recommendations for energy planning, considering the specific context and characteristics of the municipality.],

    [Interactive map interface],
    [Functional],
    [The user interface should include a map displaying assessed data points within the municipality for better spatial understanding.],

    [Conversational interface],
    [Functional],
    [The system should provide a natural language interface allowing users to interact with the AI through conversational queries.],

    [Preference memory],
    [Functional],
    [The AI should remember user preferences and past interactions to adapt responses and improve user experience over time.],

    [Multi-AI orchestration],
    [Functional],
    [The system should coordinate multiple specialized AIs through an orchestration layer to provide comprehensive energy planning assistance.],

    [Usability and accessibility],
    [Non-functional],
    [The interface should be intuitive and accessible for users with varying technical expertise.],

    [Data privacy and security],
    [Non-functional],
    [The solution must ensure the privacy and security of any non-public data, especially considering future integration of additional datasources.],

    [Extensibility and adaptability],
    [Non-functional],
    [The architecture should allow for the addition of new features and datasources as requirements evolve.],
  ),
  caption: "Requirements table",
) <requirements_table>

#highlight("TODO: ajouter nice to have multilingue, persistance, ...")

#pagebreak()
== System Design
#highlight("Renommer implémentation ?")

#highlight("Parler de tech stack, définition des données ?")

A modular, scalable and adaptable architecture was designed to ensure that the solution can adapt to evolving requirements and facilitate future enhancements. The system is organized into distinct components which are all responsible for a specific set of functionalities and ensure clear separation of concerns and ease of maintenance.

The following chapter covers both the design and the implementation aspects of the solution.

#figure(image("figs/system_design_global.svg", height: 5cm), caption: "Global system design")<global_system_design>
#highlight("TODO: mettre des numéros sur la figure?? mettre des légendes ?")
#highlight("TODO: ajouter un label aux flèches?")

The global architecture in its most simplified form is presented in the #ref(<global_system_design>) above. The system is broken down into three distinct layers:
- The frontend layer manages user interaction and presentation of data, providing an intuitive interface for users to communicate with the system.
- The backend layer is responsible for business logic, orchestrating AI agents, processing data, managing a database and handling requests from the frontend.
- The external services layer provides access to third-party Application Programming Interfaces (APIs), a set of protocols and tools that allows different software components to communicate with each other, enabling the system to retrieve data from external platforms and services.
#highlight("TODO: spécifier que database est redis ??")
The layers are made up of various components. The basic dataflow between them is presented in the #ref(<global_dataflow>) below:
#figure(
  table(
    columns: 3,
    table.header([From], [To], [Data]),
    [User (Website)], [AI Agent], [Prompt/query],
    [AI Agent (Backend)], [Local Database], [Data retrieval request],
    [AI Agent (Backend)], [Third-party APIs], [API data request],
    [Local Database], [AI Agent (Backend)], [Relevant data],
    [Third-party APIs], [AI Agent (Backend)], [API response data],
    [AI Agent (Backend)], [User (Website)], [Streamed response],
  ),
  caption: "Global system dataflow",
) <global_dataflow>

The nature and type of data that is exchanged between the AI agent and both the local database and third-party APIs will be later discussed.

Having established a high-level understanding of the system's architecture, the following sections delve into the internal details of each component, starting with the heart of the solution: the AI agent.

=== AI agent

In recent months, AI agents have gained traction with the rapid advancement of AI technologies and the increasing demand for personalized and intelligent services.
As a result, the term "AI agent" has become a buzzword, with product designers frequently applying the label to a wide range of technologies.

Most definitions agree that the key behaviour that distinguishes AI agents from other solutions is their degree of autonomy as they are able to operate and make decisions independently to achieve a set goal.
The complete solution whose sole task is urban energy planning aligns with this definition, as do each of its individual components. Therefore, the term "AI agent" will be used to describe both the entire system and its subsystems.

Human conversations rely on context and prior knowledge and so does the system's architecture. To deliver a conversational experience, it is essential that the architecture is built to effectively preserve and re-use this context throughout the discussion.
One might suggest that the straightforward approach to maintaining conversational context is to include all previous exchanges with the user. However, this method can be very inefficient and comes at great cost.

LLMs rely on a self-attention mechanism to identify and concentrate on the most relevant parts of the input sequence. Each token, the basic unit of text (word, single character, group of words...) that is processed by the model, is assigned a weight reflecting its importance. This allows the model to prioritize relevant information and ignore irrelevant details.

Hence, when the entire conversation history is provided to the model, important tokens may get lost in the larger context and potentially lead to incorrect responses (attention diffusion).

Considering this, a more efficient approach is proposed, relying on a single key assumption: each conversation focuses exclusively on energy planning for one municipality at a time.
Accordingly, the conversational context is modeled as a single object that is updated at every turn. It is defined in the codeblock #ref(<conversational_state>) :

#let destructured_state = read("code/destructured_state.py")
#figure(
  code()[
    #raw(destructured_state, lang: "python")
  ],
  caption: "Conversation state",
) <conversational_state>

#highlight("TODO: mettre au format UML??")
#highlight("TODO: mettre les sous-objets, router, ...?")
#highlight("TODO: citer pydantic, runtime, tralala???")

The different agents leverage Reasoning Language Models (RLMs), a type of LLM designed to tackle problems by breaking them into logical steps, mimicking human reasoning.
Compared to standard language models, they are particularly valuable for tasks that require logical deduction and planning but come with notable drawbacks as they are typically more computationally intensive, leading to higher operational costs and increasing latency in response times.
#highlight("TODO: citer correctement acronyme")

By constraining the conversational state and narrowing the scope of each agent, it is possible to reduce the computational load and latency by simply swapping out these large reasoning models by smaller, better-suited models.
Doing so, it becomes possible to select reasoning models that are better suited to specific tasks while reducing the computational costs.

#figure(image("figs/ai_agent_system_design.svg", height: 7.5cm), caption: "AI agent architecture")<ai_agent_design>
#highlight("TODO: ajouter flèche stream response clarificaction et answer")
#highlight("TODO: renommer critic")
#highlight("TODO: séparer proprement les API geoadmin dans un bloc external services?")
#highlight("TODO: remplacer par un schéma de FSM classique??")

The architecture in the #ref(<ai_agent_design>) above is modeled after a Finite State Machine (FSM), where each node represents an agent and each edge represents a transition that is either always executed (solid) or conditionally executed (dashed). The dynamic flow of control between agents is guided by the evolving conversational state. It is finite, per definition, as the state takes value in a discrete set.
#highlight("TODO: citer acronyme correctement?")
#highlight("TODO: enlever notion finite state machine?")

On the implementation-side, LangGraph#footnote("https://www.langchain.com/langgraph"), an open-source Python framework, is used to implement the AI agent architecture. Unlike linear pipelines, LangGraph uses a graph abstraction by default, which is particularly well-suited for this state machine architecture.
This graph-based structure brings determinism to the system’s behaviour as the flow between agents is defined by the architecture itself, rather than being dynamically determined by agent-to-agent conversations as in frameworks like Microsoft's AutoGen#footnote("https://www.microsoft.com/en-us/research/project/autogen/"). Another framework that had been considered was PydanticAI#footnote("https://ai.pydantic.dev/") which offers a structured, type-safe approach to building agent systems by leveraging Pydantic models for inter-agent communication and behaviour definitions. However, it lacks the built-in support for complex state transitions.

All of the available multi-agent AI frameworks are relatively novel and in constant evolution. LangGraph benefits from being built on top of the already renowned LangChain#footnote("https://www.langchain.com/") ecosystem which adds to its reliability and ease of integration with other technologies.
Pydantic’s type safety will still be implemented within the project to enhance data validation and error handling.

#highlight("TODO: mettre la techstack dans un chapitre différent??")

The main responsibility of each agent is as follows:
- The *Intent Router* routes the user's query to the appropriate agents and accumulates query context.
- The *Clarify Query* clarifies the user's query if it is ambiguous or incomplete.
- The *Geocontext Retriever* retrieves the geospatial data relevant to the request.
- The *Guidelines Retriever* retrieves the relevant energy planning guidelines relevant to the query.
- The *Strategy Planner* plans the energy strategy based on the retrieved data and guidelines.
- The *Critic Answer* evaluates the proposed energy planning strategy and possibly restarts the whole process.
#highlight("TODO: plus en détails ?")

With the overall solution defined, the following sections dig in the details of each agent and their implementation.

#highlight("TODO: parler tech stack et choix ollama?")

==== Intent Router <intent_router>

The intent router is a crucial component of the solution. It is the entrypoint of the system and orchestrates the different agents.

Upon receiving a user prompt, the agent analyzes the query to extract its underlying intent. This involves identifying these elements:
- The intent: specifies whether the query is "factual" (for e.g. requesting data) or "actionable" (seeking planning guidance, recommendations, or strategic advice).
- The location: the municipality name mentioned in the user request, if available.
- The aggregated query: a summary that combines all available context from the current conversation and the previous query into a single one.
- The conversation type: identifies the conversational context ; "new_analysis" (fresh query), "correction_request" (user questions the accuracy of a previous response) or 'follow_up' (user requests additional detail or expansion on the same topic).
- The need for clarification: defines whether more information is needed to understand what the user wants (e.g., missing location, unclear intent, or vague request).
- The needs for memoization: specifies if the user provided explicit preferences, corrections to assumptions, or scope refinements that should be remembered for future queries (for e.g. the format used to summarize the retrieved data or only considering a single aspect from certain datapoints).

Implementation-wise, the output of the language model is constrained to a single Pydantic data schema, _RouterOutput_.
While these models typically generate natural language responses, these complex multi-agent systems benefit from a structured output format that can easily be further processed.
This is possible thanks to OpenAI#footnote("https://openai.com/"), introducing the support for structured outputs in late 2024, a feature that has since been adopted by many providers.
#highlight("TODO: inclure prompt system")

All these fields except the aggregated query take value in a finite set of options (considering the _location_ field is either set or unset.).
Consequently, it is very easy to plan and orchestrate the following actions.

Since the application must offer a conversational experience, the previous _RouterOutput_ is accumulated on every turn.
Past fields are only updated if they differ from the new ones. As such, context and knowledge is properly accumulated over time.
Once the municipality is provided, for example, it is not needed anymore as the request is assumed to concern the same municipality.

On the other hand, when a request treats a different municipality, both the _context_tools_ and _context_constraints_ defined in #ref(<conversational_state>) are reset. This way, the data associated with the previously discussed municipality is cleared.
To ensure correct implementation, whenever a request concerns a different municipality, both the _context_tools_ and _context_constraints_ defined in #ref(<conversational_state>) are reset. This means the geospatial data and guidelines previously associated with the conversation are cleared.

On top of that, user-provided feedback and corrections shape the system's behaviour allowing it to adapt to the user's preferences.
When there is a need for memoization, the system stores both the previous query (the _corrected_) and the current query (the _correctee_) in the user's namespace, in the database.

#highlight("TODO: parler que store parallèle ??")
#highlight("TODO: parler quelque part d'async??")
#highlight("TODO: schéma avec flux entiers, database, ...")
#highlight("TODO: décire que local database redis...")
#highlight("TODO: mettre au format UML??")
#highlight("TODO: parler au dessus coûte cher de faire un appel LLM")

An assumption still lies in the nature of the field _location_ as it is assumed to either be set or unset. A set location does not necessarily imply that it is a valid municipality, inscribed in the published Swiss official commune register#footnote("https://www.bfs.admin.ch/bfs/en/home/basics/swiss-official-commune-register.html").
A solution is proposed in the section #ref(<geocontext_retriever>, supplement: it => it.body).

#highlight(
  "TODO: dire que utiliser un dictionnaire difficile car mises à jour fréquentes par exemple ici ou plus bas ??",
)

Finally, the query is routed according to the #ref(<ai_agent_design>):
- If clarification is needed (either because the need for clarification is explicitly requested, or fields are missing), the request is sent to the clarify query agent (2).
- If the conversation type is a correction request, the query is sent to the geocontext retriever agent (4).
- If the intent is said to be actionable, the request is sent to both the geocontext retriever and the guidelines retriever agents, concurrently - (4) and (5).
- Otherwise, the query is sent to the geocontext retriever agent (4).
#highlight("TODO: mieux expliquer routing et pourquoi comme ça")
With the aim of the user's query now clearly defined, the next step is to address any ambiguities or missing information with the clarification agent.

==== Clarify Query

Clarifying and resolving vagueness in the user's query is essential to better understand the fundamental intent and provide an aligned response.

With the output of the intent router agent properly defined, the two cases which lead to the need for clarification are either an explicit request for clarification due to ambiguity or missing information.

Those two cases are both handled at once as a language model is prompted with the user's query and missing fields to generate and stream a response inquiring for further information or clarification (#ref(<ai_agent_design>), transition 3).
In the following turn, the newly provided information is merged with the previously deduced intent as designed and presented in the section #ref(<intent_router>, supplement: it => it.body).

With no _structural_ ambiguity left in the user's query, the intent router agent can now proceed to route the query to the *Geocontext Retriever* and *Guidelines Retriever* agents.

#highlight("TODO: filer le prompt")

==== Geocontext Retriever <geocontext_retriever>

Energy planning as defined in the solution requires assessing the energy resources, infrastructure, potential and needs within the municipality. The geocontext retriever is responsible for this task.

Before profiling the municipality, it is essential to identify the different public datasources that are available.
Throughout its federal and cantonal institutions, Switzerland provides a wide range of public data such as GeoAdmin#footnote("geo.admin.ch"), the geographic information platform of the Confederation which offers direct access to geospatial data and maps.

The data originates from various offices commissioned by the Confederation:
- Swiss Federal Office of Energy (SFOE)
- Federal Office for Spatial Development (ARE)
- Federal Office of Topography (swisstopo)
- Federal Office for Agriculture (FOAG)
- Federal Office for the Environment (FOEN)

The GeoAdmin API#footnote("https://api3.geo.admin.ch/index.html") provides a standardized interface for querying and manipulating geospatial data and relies on fair usage policies (20 requests per minute on a 24/7 average).
The datasets are also available for download.

The choice has been made to use the GeoAdmin API instead of downloading and maintaining local datasets as it ensures (1) that the data is always up to date and (2) removes the need for additional setup and maintenance of a dedicated geospatial database, a task that is particularly time-consuming in such a short time frame.

In a real-world scenario, exploiting data locally allows for preprocessing and aggregation which significantly reduces latency during user interactions.
Mechanisms such as caching and geospatial indexing would be useful for greater scalability of the solution.
#highlight("TODO: enlever indexing avant la suite?")

Datasets are often labeled as layers, as the data is organized according to the geospatial paradigm. Data is discretized into points, meshes, polygons and other spatial representations, all defined as a feature. Those features are independent geometries located in the space, without inherent relationships.
Thus, identifying relevant features within a municipality implies searching them inside its geographic boundaries, since no relation lies between these entities.

Although the GeoAdmin API enables searching for features in a given area, it is subject to a maximum number of 50 features retrieved per request.
Consequently, identifying them requires breaking down the search area into smaller sub-areas and querying each sub-area separately.

This has been implemented by first clipping settlements and centres of larger cities onto the municipality's geometry, optimizing the search area, and applying a spatial tiling on top. Different layers obviously require different tiling sizes, depending on the number and resolution of features.

The #ref(<datasets_table>) presents the data sources incorporated in the solution:

#set table(fill: (x, y) => if calc.odd(y) and x != 0 { rgb("EAF2F5") })
#figure(
  rotate(-90deg, reflow: true, table(
    columns: 5,
    table.header([Category], [Layer ID], [Description], [Unit], [Discretization]),
    [*Needs*],
    [ch.bfe.fernwaerme-nachfrage_industrie],
    [Heat and cooling demand from industry],
    [MWh/year],
    [100m x 100m],

    [],
    [ch.bfe.fernwaerme-nachfrage_wohn_dienstleistungsgebaeude],
    [Heat and cooling demand from residential and commercial buildings],
    [MWh/year],
    [100m x 100m],

    [], [ch.bafu.klima-co2_ausstoss_gebaeude], [Greenhouse gas emissions from buildings], [kg/m²], [Per building],

    [*Potential*],
    [ch.bfe.kleinwasserkraftpotentiale],
    [Potential of small hydropower plants],
    [kW/m],
    [Per watercourse],

    [], [ch.bfe.waermepotential-gewaesser], [Potential heat use of water bodies], [GWh/year], [Per water body],

    [],
    [ch.bfe.solarenergie-eignung-daecher],
    [Suitability of roofs for use of solar energy],
    [kWh/year],
    [Per roof pane],

    [], [ch.bfe.solarenergie-eignung-fassaden], [Solar energy: suitability of façade], [kWh/year], [Per façade],

    [], [ch.bfe.biomasse-nicht-verholzt], [Biomass potential], [TJ], [Per municipality],

    [], [ch.bfe.fernwaerme-angebot], [Potential heat recovery from WWTPs], [MWh/year], [Per plant],

    [*Infrastructure*], [ch.bfe.statistik-wasserkraftanlagen], [Hydropower plants: statistics], [GWh/year], [Per plant],

    [], [ch.bfe.windenergieanlagen], [Wind energy plants], [GWh/year], [Per turbine],

    [], [ch.bfe.biogasanlagen], [Biogas plants], [kWh/year], [Per plant],
    [], [ch.bfe.kehrichtverbrennungsanlagen], [Waste incineration plants], [MWh/year], [Per plant],

    [], [ch.bfe.elektrizitaetsproduktionsanlagen], [Electricity production plants], [kW], [Per plant],

    [], [ch.bfe.thermische-netze], [Thermal networks], [MWh/year], [Per network],

    // [], [ch.swisstopo.swissboundaries3d-gemeinde-flaeche.fill], [Municipalities], [-], [Per municipality],

    // [], [ch.swisstopo.vec200-landcover], [Swiss land cover], [-], [Per surface],
  )),
  caption: "Public datasets",
) <datasets_table>

#highlight("TODO: mieux décrire ce qu'il y a dans le tableau?")
#highlight("TODO: bouger label figure au dessus?")

The discretization of the different datasources showcases the importance of spatial tiling when dealing with this data.

In the average municipality, certain features are few and easily assessable whereas it is impossible to retrieve meaningful insights from the greater-resolution datasets (for e.g. the suitability of roofs, per roof pane) without additional processing or aggregation.

Therefore, the features within the municipality are (1) identified within the municipality, (2) aggregated to the municipality level, and (3) brought back to the same GWh/year unit#footnote("Only applies to energy measures. Energy is deduced from power, in watts, assuming non-stop operation (24/7/365)."). This standardization allows for an easier and consistent comparison of scalars, on a yearly basis, crucial for interpretability but comes with drawbacks:
- The information of variability within the municipality itself is lost.
- The aggregation of data is a costly process, especially when doing this on the fly.

The first issue is partially recovered from the fact that the layers are displayed at their original resolution, in the web interface. This way, the variability can be easily visualized.

The second issue is mitigated depending on the nature of the data. The basic approach to aggregation requires the summation of values and is only needed for datasets that benefit from great precision. On the other hand, datasets that do not require such precision are subject to statistical estimation:
- The spatial tiling is randomly sampled.
- The features within the sampled tiling are identified and their values summed.
- The sample mean and standard deviation are calculated.
- The confidence interval is computed using a T-distribution and confidence level.
#highlight("TOOD: mettre réf. mathématique??")
Choosing the sampling size and confidence level is important for a proper statistical estimation. In this work, both parameters are set empirically and kept relatively large to benefit from lower computational costs, but without optimizing for the best possible accuracy.
As such, only the suitability of roofs and façades for use of solar energy is estimated using this technique as they are both datasets which showcase potential of exploitation rather than precise measurements and are well distributed in the geographic space.

#highlight("TODO: ajoute schéma sampling statistique")
#highlight("TODO: parler ajouter des modèles simu et autre dans les tools?")

With the data standardized and properly aggregated, the geocontext retriever agent must now be able to interact with it.

Previously, AI agents were described as autonomous systems able to operate and make decisions independently. These operations rely on tools.

A construction worker, for example, has different tools for different needs, such as a hammer for nails, a saw for cutting wood, and a level for ensuring straight walls. The tools come with a set of instructions describing how to use them and what to expect from them.

The same applies to AI agents, which have, in this work, different tools allowing them to query, aggregate and retrieve data from the datasets in #ref(<datasets_table>).
Consequently, language models can leverage their natural language processing capabilities to choose the appropriate tools for the query.

An issue with the current approach is that when the _toolbox_ is too large, it becomes difficult for the language model to choose the right tools for the job.
This issue is addressed by exploiting the power of embeddings.

An embedding is a mathematical representation of data in a high-dimensional vector space where semantically similar information are mapped to nearby points.
This enables the system to embed the descriptions of the different tools and easily retrieve them semantically. On top of that, it is more efficient computation-wise than prompting the language model to choose them.
#highlight("TODO: définition terminologie? prompting")
#highlight("TODO: parler de reranking?")

When retrieving tools, the system computes the cosine similarity between both embeddings to quantify the semantic similarity.
Finally, the quartile coefficient of dispersion is measured against the distribution of retrieved scores. This indicator provides a measure of the uniformity of the retrieved tools.
As such, uniform tools are provided to a language model, which is then prompted to choose the appropriate ones.

This approach reduces the overall computational cost while increasing the quality of tool selection.

#highlight("TODO: référencer cosine, bibliographie")
#highlight("TODO: référencer coefficient correctement, bibliographie")

#highlight("TODO: faire un schéma détaillé du processus de l'agent?")

With the appropriate tools chosen, the system can effectively retrieve the data. It is simply added to the _context_tools_ field in the conversational state (#ref(<conversational_state>)).
Geospatial information is accumulated over the conversation turns, allowing for context-aware planning and consistent, spatially informed decisions. It is only reset when switching to a new municipality as it becomes invalid.

In the section #ref(<intent_router>, supplement: it => it.body), the validity of the location is not confirmed. This is directly implemented in the different tools above and routing of this agent (#ref(<ai_agent_design>)):
- If the location is non-valid, retrieving data raises an error and the request is routed to the clarify query agent.
- Otherwise, the query is sent to the strategy planner agent (6).
#highlight("TODO: manque flèche vers clarification")

Once the relevant data is gathered, the next stage is for the strategy planner agent to analyze this information to conduct proper planning.

#highlight("TODO: citer external services database")
#highlight("TODO: mettre un schéma du tiling?")
#highlight("TODO: dire requêtes API concurrent")
#highlight("TODO: parler spécifique map et système coordonnées?")
#highlight("TODO: ajouter tool energy needs, heuristique et détailler planification énergétique?")

==== Guidelines Retriever

The sole difference between enumerating the data, as collected in the geocontext retriever, and proper energy planning lies in the measures that are taken in response to identified issues. Those measures are conditioned by guidelines, broken down into multiple sources:

The primary document called _Vision 2060 et objectifs 2035_ has been adopted in 2019 and sets intermediate targets for 2035 that take into account the energetical landscape of Valais/Wallis, current knowledge, as well as federal energy and climate policies with the ultimate goal of achieving a 100% renewable and indigenous energy supply in 2060.

Moreover, the _Plan directeur 2019_ adopted by the federal council on the 1st of May 2019, states the strategy for the canton's territorial development in the form of 49 information sheets, distributed across the five activity sectors: _Agriculture, forest, landscape and nature_ (1), _Tourism and leisure_ (2), _Urbanization_ (3), _Mobility and transport infrastructure_ (4) and _Supply and other infrastructure_ (5).

Finally, the legal framework is defined by two key legislative documents. Notably, the _RS 705.1 - Loi sur les constructions (LC)_ establishes the regulations for construction activities, while the _RS 730.1 - Loi sur l'énergie (LcEne)_ defines the objectives and requirements for sustainable energy supply.
#highlight("TODO: citer autrement -> bib?")

These documents are specifically designed and structured to convey information to the public and come in a single Portable Document Format (PDF) and are available in both french and german. They are organized into sections, subsections or paragraphs which reference figures, tables, plots, past paragraphs and so on.

Visual structure does not necessarily imply a logical flow of information. A document can look and feel organized but still lack a proper machine readable structure.
In practice, it is neither realistic nor scalable to expect a human to manually extract all the key information needed for energy planning from such complex documents. Therefore, it becomes essential to delegate this task to the computer, enabling automated extraction and processing of documents.

When data lacks clear structure, it becomes difficult to extract information using algorithms or systematic procedures. However, advances in Multimodal Large Language Models (MLLMs) offer a solution as these models are designed to process and understand information presented in various modalities such as text, images, audio, and video. Paired with existing methods that are able to extract raw text from these documents, it has become easier to extract precise information from visually organized and heterogeneous documents by understanding not only the way information is displayed but also its underlying semantic meaning.

As such, a systematic approach is applied when extracting information from these documents:
- Raw text is extracted from the documents on a per-page basis.
- Each page is rendered into an image.
- Each rendered page and associated text are processed using MLLMs to retrieve key insights and interpret the information within the page.
#highlight("TODO: citer prompt")

The extracted information is formatted in markdown, utilizing headings to structure the summary. It is then broken down into smaller chunks, each chunk being a "chapter" derived from the markdown content. Since only individual chunks are considered in subsequent steps, there is no need to perform an analysis across neighboring pages to ensure that the information is retrieved from its full context.

Finally, the extracted information from each page is encoded into an embedding and stored in the local database, along with its associated chunks and metadata.

With clear guidelines extracted from documents in any format, it is necessary to identify those that are relevant to the user's query. Since those are already embedded, the related guidelines are simply those that are closest to the embedded request, as described in the #ref(<geocontext_retriever>, supplement: it => it.body) section is applied.
#highlight("TODO: revoir déf?")

An issue still lies in how the guidelines themselves are _designed_. While the objectives and figures outlined inside these documents concern the entire canton, this solution is only scoped to municipalities.
Therefore, these quantitative targets must be scaled down to reflect the municipality’s specific context and expectations.
#highlight("TODO: mettre un exemple?")

Identifying these key figures which need rescaling is not a simple task as broader context is required to assess that.
In order to achieve this, a language model is prompted with the task of identifying key figures that need rescaling.
#highlight("TODO: citer prompt")

Finally, they are multiplied by a factor corresponding to the ratio of the municipality's number of residents to the total population of the canton.
While this is a straightforward way to scale targets, proper rescaling should take into account the economic activity, energy landscape and industrial presence in the municipality to ensure a more accurate adjustment.

The adjusted guidelines are accumulated onto the _context_constraints_ field in the conversational state (#ref(<conversational_state>)).
Similarly to the geospatial information described in the #ref(<geocontext_retriever>, supplement: it => it.body) section, the processed guidelines are accumulated in the state as the conversation goes on and only cleared when switching to a new municipality.

With the relevant guidelines retrieved and rescaled, the query is routed to the strategy planner agent (#ref(<ai_agent_design>), transition 6) which will use them as clear constraints.

#highlight("TODO: mettre un premier chapitre preprocessing?")
#highlight("TODO: cite plus tard que Retrieve multiple chunks recreate the context")
#highlight("TODO: reformuler et enlever les formulations 'we' ?")
#highlight("TODO: dire que je traduis en anglais ou pas, pff")
#highlight("TODO: citer modèle utilisé MLLM et embeddings ?")
#highlight("TODO: citer comment traiter données communales")
#highlight("TODO: inclure prompts dans bibliographie")

==== Strategy Planner <strategy_planner>

At this stage, every bit of information that is needed to establish a proper energy planning strategy is gathered into the conversational context (#ref(<conversational_state>)).
The user's prompt has been broken down and analyzed with relevant data points and guidelines retrieved and processed.

In the #ref(<intent_router>, supplement: it => it.body) section, the _intent_ field is defined to either be factual (requesting data) or actionable (seeking planning guidance, recommendations, or strategic advice).
This distinction is crucial as it allows the agent to differentiate between the two tasks, the latter being more expensive because of the extra complexity of correlating guidelines and data to concretize a strategy.

Factual queries still contribute to the final goal of establishing an energy planning strategy as it enables users to assess the profile of the municipality and subsequently refine and guide the system into a more effective and informed strategy.

As defined in the same section, the local database stores user feedback and corrections to past queries.
The agent retrieves pertinent preferences and memories related to the current query and shapes the response according to those expectations.
Like tools and guidelines, memories are stored as embeddings and are therefore retrieved based on semantic similarity.

Finally, similar tools to those retrieved by the geocontext retriever agent are retrieved in order to generate tailored recommendations.
The selection of similar tools is based on their categorization, as defined in #ref(<datasets_table>). This encourages assessing the full spectrum of available data for any municipality.
#highlight("TODO: inclure prompts actionable et factuel et expliquer comment leverage les guidelines")

The language model response, which is the answer to the user's query, is streamed to the web interface.

While the user examines the response, it is sent to critic agent (#ref(<ai_agent_design>), transition 8), which will evaluate its quality and act accordingly.

==== Critic

One of the main challenges in designing a conversational AI solution for real-world problems is ensuring the accuracy and relevancy of the response.
Decomposing the complex task of energy planning into smaller steps, each addressed by a dedicated agent, allows for specialized solutions, leading to more precise and context-aware interactions.

Inaccurate responses issue from different factors such as incorrect data or lack of context.
In this work, an element that may be even more impactful is the interpretation of this same context, curated by the different agents involved in the workflow.
These interpretations errors typically include:
- Mathematical errors where data points are added or subtracted to support insights.
- Flawed conclusions from different metrics incorrectly treated or incorrect assumptions.

As such, a language model is prompted the response generated in the <strategy_planner> with the data points and guidelines that shaped it.
Its output is a boolean value that indicates whether the response has been interpreted correctly based on the rules above.
#highlight("TODO: insérer prompt")

If it the response is not satisfactory, the complete process is restarted as if the user had just prompted the system (#ref(<ai_agent_design>), transition 9).
A maximum of three attempts are allowed before the workflow is not restarted anymore.

At this point, the user's request has been answered and the system is ready to receive a new request, refining the proposed energy planning.

This concludes the design and implementation of the AI agent responsible for energy planning.

=== Web Interface

Designing and providing an interface that is user-friendly and convenient for the user to interact with the AI system is key to the adoption of the product.
The exactitude and accuracy of the AI system weighs heavily on the user's satisfaction but so does the user experience and presentation that is offered.

In recent times, the tendency shifted from traditional desktop applications to web-based interfaces, offering greater accessibility for portable devices.

On the development side, the interface was implemented using React#footnote("https://react.dev/"), a popular JavaScript framework released by Facebook (now Meta) in 2013. React offers a declarative and efficient way to build user interfaces, offering a clean and modular approach using components.

In reality, the choice of framework is not particularly critical in this context. Dozens of frameworks claim to revolutionize the way developers build web applications, but all lead to similar outcomes despite different approaches and philosophies. Past experiences with React and its ecosystem made it a comfortable and efficient choice for this project.

#highlight("TODO: parler vite, bun et docker???")

What is more important is paradigm of presentation is driven

// parler pas de support mobile
// parler techstack

// overview
// ux
// sse
// map
// faire un graphe des événements ?
// concept de persistance
// mise en évidence datasources augmente confiance
// heurisitque consommation
#highlight("TODO: mettre des screenshots?")

=== Limitations
#highlight("TODO: en faire un chapitre par composant au dessus ou en dehors de la partie méthodologie?")

// CITER DES EXAMPLES ICI MGL
// llm incapable de faire des maths
// données privées ?
// interprétation données
// qualité des données
// memories et application format
// interprétation données petits modèles
// fuzzy search <- amélioration plutôt que limitation

= Results

// citer date version code utilisée pour comparer expert et llm
// parler difficulté llm assigner un score, alors tabelle prédéfinie
// impossible de comparer les résultats avec état de l'art car pas même critères

= Discussion

= Conclusion
// amélioration graphe
// MCP
// train classificateur guidelines, fine tune, ...
// train classificateur intent

#pagebreak()
#heavy-title(i18n(doc_language, "bibliography-title"), mult: 1, top: 0.5em, bottom: 0.3em)
// generate bib file RIS script (https://www.bruot.org/ris2bib/)
#bibliography("bibliography.bib", full: true, style: "ieee", title: none)
#highlight("TODO: mettre une deuxième bib. technique uniquement!!")
// #bibliography(("bibliography.bib", "technical_reference.bib"), full: true, style: "ieee", title: none)

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
