#import "@preview/isc-hei-bthesis:0.5.3": *

#let doc_language = "en" // Valid values are en, fr

#show: project.with(
  title: "Optimisation de la planification énergétique urbaine par l'orchestration de l'IA",
  sub-title: "A Bachelor Thesis in Data Engineering", // Optional

  is-thesis: true,
  split-chapters: true,

  thesis-supervisor: "Prof. Jessen Page",
  thesis-co-supervisor: "Florian Desmons", // Optional, use none if not needed
  thesis-expert: "Nils Schüler", // Optional, use none if not needed
  thesis-id: "ISC-ID-2501", // Your thesis ID (from the official project description) or none if not used
  project-repos: "https://github.com/dij0s/PETAL", // Your project repository


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
#highlight("TODO: check ortographe")
#highlight("TODO: update le thesis-id")
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

The abstract should be self-contained, clear and usually does not exceed 250-300 words. It allows readers to quickly understand the purpose and outcomes of the thesis without reading the full document.

The abstract *must* be written in both French and English.

Please also insert your project git/github URL HERE if your project is not confidential.

#lorem(150)

#abstract-footer("en")

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

#abstract-footer("fr")

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
// Enable headers and footers from this point on
#set-header-footer(true)

= Introduction

#highlight("TODO: diviser en sections? contexte, problématique...")

Over the past few decades, society has been sensitized and slowly became more aware of significant problems that we are likely to face in the coming years.

Climate change and other environmental issues arise as a result of human-driven activities.

Scientists have monitored this matter and proposed various frameworks to address and mitigate these problems. In Switzerland, these different frameworks are implemented in the legislation and guidelines (at federal and canton levels) to steer the country towards a more sustainable future.
Municipalities may introduce in their regulations energy requirements that are more constraining than those set by the cantonal law as per article 12, al. 5 of #ref(<rdb6SC4eJbME>).

== Context

Municipalities in Switzerland are required to submit an energy planning document which outlines their future strategies to comply with those directives while also considering the characteristics of their energetical landscape.

These different properties can be quantified and analyzed through the use of a very valuable resource: data.
Data is emitted by various sources ; sensors, energy models or citizens' records for example all yield data points that help us assess different indicators we are willing to measure against our municipality. These indicators, _in fine_, help us evaluate our progress towards energy-related goals.

Over the past few years, #acr("AI") has rapidly transformed our habits when interacting with information.

== Problem

#acrpl("LLM") allow users to interact with these systems in natural language facilitating the interface between humans and _machines_. They can provide insights into vast amounts of data at speed and scales which are beyond our capabilities.

This work tackles this exact problem that is the implementation of a solution which assists users into energy planning for a municipality.

This complex problem is approached by leveraging the power of _specialized_ AIs which each offer expertise into a variety of domains and are coordinated using an _orchestration_ AI to provide a solution. The expertise of the user interacting with the system tailors this solution to the specificities of the municipality.

The key steps in the engineering of this implementation are identifying the key information and datasources that are relevant to this process, structure the different AIs into an architecture whose components and interfaces are well-defined and ultimately implementing the solution.

== Objectives

The main objective of this work is to investigate how effective and reliable such a solution is and to assess its strengths and weaknesses. Additional goals are also outlined:
- Defining the important information in assisting user decision making.
- Understanding which datasources are available and relevant for this decision making.
- Structuring the decision by using an orchestration AI and many specialized AIs.
- Training specialized AIs on specialized datasets.
- Simplifying the user interface by handling the communication between the orchestration AI and the specialized AIs.

The project is scoped to municipalities within the canton of Valais/Wallis and strictly relies on publicly available data.
Certain measures are taken to ensure the privacy and security of data that would not be of public order considering the future implementation of extra datasources.

The solution is designed to offer a user-friendly interface from which users can interact with the system in a conversational manner and visualize a map of the municipality with different layers.

User behavior is analyzed to takeaway user preferences which gradually adapt the answers to better meet the user expectations.

#highlight(
  "TODO: parler de la structure du document et référencer methodologie?",
)

= Artificial Intelligence notice

While generative artificial intelligence is the core of this work, it has also been a great help in assisting the following tasks: prompt engineering (1) and thesis report rewording (2)#footnote("The language models used for these tasks are GPT-4.1 (OpenAI) and Claude 3.7 (Anthropic).").

The different prompts defined in the appendix were refined with the use of generative artificial intelligence. This strongly enhances the quality of the prompts, implementing prompt engineering techniques to ensure clear, concise and effective instructions.

#ref(<prompt_creation>) defines typical directives that were given to the language model.

On top of that, sentences in the following report were often revised with the assistance of a language model, improving their clarity and readability.

Besides that, all other aspects of this work are my own.

= State of the Art <state_of_art>

#highlight("TODO: CHECKER VIM TEMP!!")
#highlight("TODO: parler état de l'art LLM tout court et décrire la technologie ou juste approcher")

Large language models are highly effective tools for natural language processing and offer various opportunities to enhance our day-to-day tasks and workflows.
Ever since they have been introduced to the public, they have been adopted across a wide range of fields and applications.

The AI Institute at ITMO University published a paper in 2025 titled _LLM Agents for Smart City Management: Enhancing Decision Support Through Multi-Agent AI Systems_ #ref(<kalyuzhnayaLLMAgentsSmart2025>). The study examines how the natural language processing strengths of LLMs, combined with the distributed problem-solving abilities of multi-agent systems, can enhance urban decision-making processes.

The research focused on the testing of three hypotheses: (1) evaluating the capability of LLM agents to effectively route and process diverse urban queries against existing urban information systems, (2) the effectiveness of #acr("RAG") technology in improving response accuracy when working with local knowledge and regulations and (3) the impact of integrating LLM agents with existing urban information-systems - increasing efficiency and decreasing the decision making process time.

LLM agents (sometimes called AI agents) are software systems that use AI to pursue goals and complete tasks on behalf of users. They show reasoning, planning and memory and have a level of autonomy to make decisions, learn and adapt as per #link("https://cloud.google.com/discover/what-are-ai-agents?hl=en")[cloud.google.com].

Their proposed solution was tested against 150 question-answer pairs and used St. Petersburg's Digital Urban Platform as a testbed.
The testing dataset was curated and built by a group of human experts such as specialists in urban data analysis, GIS specialists and urban architects.

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

As we forget about the use case and consider more generic research on the matter, we encounter  publications that rather focus on the optimization of various aspects of AI agents, such as their scalability, efficiency and robustness. While this work tackles some of these challenges, it does not incorporate major research efforts into these areas.

Another AI-centric solution relevant to this use case is aino#footnote("https://www.aino.world/"), described on their homepage as an _AI GIS Analyst for Urban planning teams_. Developed and marketed in the United States, it is a commercial business solution offering a platform where an AI analyzes sites and provides visual insights from simple questions. This solution inspired me into a few design and user experience improvements in my work as no implementation details are provided.

These two points of view position this work within the fast-changing field of AI-driven solutions for urban planning and beyond.

= Methodology <methodology>

The following chapter provides an overview of the methodology that has been adopted in this work and describes the approach taken to design, implement and further evaluate the proposed solution.
A clear emphasis is put onto the key decisions that have shaped the solution throughout its development as this chapter aims to offer full transparency and reproducibility which enables readers to understand the rationale and structure behind it.

== Requirements

Identifying and describing the primary requirements lays the groundwork for the design of the solution.
The main requirements were established and further refined as the project progressed to meet the expectations and project goals throughout ongoing discussions with the supervisors.
Those are categorized into functional and non-functional requirements. Functional requirements focus on user requirements and product features whereas non-functional requirements focus on user expectations and product properties.

The first step is to understand the problem that is solved. Energy planning typically involves:
- Identifying available energy resources#footnote("The resources and needs are assessed within the geographical boundaries of the municipality, only."), infrastructure and untapped potential.
- Characterizing the needs.
- Assessing different measures and their impacts ; sobriety (reducing energy consumption), efficiency (more efficient technologies) and production of renewable energy sources.

Hence, assisting users in energy planning requires a solution that can gather relevant data sources, analyze and present the current energy landscape of the municipality and provide actionable recommendations tailored to the context of the municipality while ensuring compliance with the legislation and guidelines that apply to said municipality.

Besides that, it had been requested that the user interface has a map showcasing the assessed data points within the municipality as well as for the AI to be able to _remember_ the user's preferences and past interactions to have the answer better fit what the user expects.

These initial requirements established the basis for the project. As the solution was evaluated with supervisors on a weekly basis, additional requirements emerged gradually shaping the solution to enhance the overall solution and meet the needs of energy planning.
#highlight("TODO: revoir la dernière phrase, ajouter des autres requirements ?")

#show table.cell.where(y: 0): strong
#show figure: set block(breakable: true)
#set table(stroke: (x, y) => if y == 0 {
  (bottom: 0.7pt + black)
})
#set table(
  fill: (x, y) => { if calc.odd(y) { rgb("F7F9FA") } },
  align: (x, _) => if x == 0 { center } else { left },
)
These requirements are summarized in the #ref(<requirements_table>) below:
#figure(
  table(
    columns: (auto, 30%, auto),
    table.header([Type], [Requirement], [Description]),
    [Functional],
    [Gather relevant data sources],
    [The solution must collect and integrate data from various sources relevant to the municipality's energy landscape.],

    [Functional],
    [Analyze and present energy landscape],
    [The system should process and visualize the current energy situation of the municipality, enabling users to understand available resources and needs.],

    [Functional],
    [Provide actionable recommendations],
    [The solution must generate tailored recommendations for energy planning, considering the specific context and characteristics of the municipality.],

    [Functional],
    [Interactive map interface],
    [The user interface should include a map displaying assessed data points within the municipality for better spatial understanding.],

    [Functional],
    [Conversational interface],
    [The system should provide a natural language interface allowing users to interact with the AI through conversational queries.],

    [Functional],
    [Preference memory],
    [The AI should remember user preferences and past interactions to adapt responses and improve user experience over time.],

    [Functional],
    [Multi-AI orchestration],
    [The system should coordinate multiple specialized AIs through an orchestration layer to provide comprehensive energy planning assistance.],

    [Non-functional],
    [Usability and accessibility],
    [The interface should be intuitive and accessible for users with varying technical expertise.],

    [Non-functional],
    [Data privacy and security],
    [The solution must ensure the privacy and security of any non-public data, especially considering future integration of additional datasources.],

    [Non-functional],
    [Extensibility and adaptability],
    [The architecture should allow for the addition of new features and datasources as requirements evolve.],
  ),
  caption: "Requirements table",
) <requirements_table>

#highlight("TODO: ajouter nice to have multilingue, persistance, ...")

#pagebreak()
== System Design <system_design>

A modular, scalable and adaptable architecture was designed to ensure that the solution can adapt to evolving requirements and facilitate future enhancements. The system is organized into distinct components which are all responsible for a specific set of functionalities and ensure clear separation of concerns and ease of maintenance.

The following chapter covers both the design and the implementation aspects of the solution.

#figure(image("figs/system_design_global.svg", height: 5cm), caption: "Global system design")<global_system_design>

The global architecture in its most simplified form is presented in the #ref(<global_system_design>) above. The system is broken down into three distinct layers:
- The frontend layer manages user interaction and presentation of data, providing an intuitive interface for users to communicate with the system.
- The backend layer is responsible for business logic, orchestrating AI agents, processing data, managing a database and handling requests from the frontend.
- The external services layer provides access to third-party #acrpl("API"), a set of protocols and tools that allows different software components to communicate with each other, enabling the system to retrieve data from external platforms and services.
#highlight("TODO: spécifier que database est redis ??")
The layers are made up of various components. The basic dataflow between them is presented in the #ref(<global_dataflow>) below:
#figure(
  table(
    columns: 3,
    align: center,
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

Most definitions agree that the key behavior that distinguishes AI agents from other solutions is their degree of autonomy as they are able to operate and make decisions independently to achieve a set goal.
The complete solution whose sole task is urban energy planning aligns with this definition, as do each of its individual components. Therefore, the term "AI agent" will be used to describe both the entire system and its subsystems.

Human conversations rely on context and prior knowledge and so does the system's architecture. To deliver a conversational experience, it is essential that the architecture is built to effectively preserve and re-use this context throughout the discussion.
One might suggest that the straightforward approach to maintaining conversational context is to include all previous exchanges with the user. However, this method can be very inefficient and comes at great cost.

LLMs rely on a self-attention mechanism to identify and concentrate on the most relevant parts of the input sequence. Each token, the basic unit of text (word, single character, group of words...) that is processed by the model, is assigned a weight reflecting its importance. This allows the model to prioritize relevant information and ignore irrelevant details.

Hence, when the entire conversation history is provided to the model, important tokens may get lost in the larger context and potentially lead to incorrect responses (phenomenon called attention diffusion).

Considering this, a more efficient approach is proposed, relying on a single key assumption: each conversation focuses exclusively on energy planning for one municipality at a time.
Accordingly, the conversational context is modeled as a single object that is updated at every turn. It is defined in the #ref(<conversational_state>) below:
#let destructured_state = read("code/destructured_state.py")
#figure(
  code()[
    #raw(destructured_state, lang: "python")
  ],
  caption: "Conversation state",
) <conversational_state>

#highlight("TODO: PARLER OLLAMA MGL")

The different agents leverage #acrpl("RLM"), a type of LLM designed to tackle problems by breaking them into logical steps, mimicking human reasoning.
Compared to standard language models, they are particularly valuable for tasks that require logical deduction and planning but come with notable drawbacks as they are typically more computationally intensive, leading to higher operational costs and increasing latency in response times.

By constraining the conversational state and narrowing the scope of each agent, it is possible to reduce the computational load and latency by simply swapping out these large reasoning models by smaller, better-suited models.
Doing so, it becomes possible to select reasoning models that are better suited to specific tasks while reducing the computational costs.

#figure(image("figs/ai_agent_system_design.svg", width: 100%), caption: "AI agent architecture")<ai_agent_design>
#highlight("TODO: vérifier partout que les réf. transitions sont bonnes")

The architecture in the #ref(<ai_agent_design>) above is modeled after a #acr("FSM"), where each node represents an agent and each edge represents a transition that is either always executed (solid) or conditionally executed (dashed). The dynamic flow of control between agents is guided by the evolving conversational state. It is finite, per definition, as the state takes value in a discrete set.

On the implementation-side, LangGraph#footnote("https://www.langchain.com/langgraph"), an open-source Python framework, is used to implement the AI agent architecture. Unlike linear pipelines, LangGraph uses a graph abstraction by default, which is particularly well-suited for this state machine architecture.
This graph-based structure brings determinism to the system’s behavior as the flow between agents is defined by the architecture itself, rather than being dynamically determined by agent-to-agent conversations as in frameworks like Microsoft's AutoGen#footnote("https://www.microsoft.com/en-us/research/project/autogen/"). Another framework that had been considered was PydanticAI#footnote("https://ai.pydantic.dev/") which offers a structured, type-safe approach to building agent systems by leveraging Pydantic models for inter-agent communication and behavior definitions. However, it lacks the built-in support for complex state transitions.

All of the available multi-agent AI frameworks are relatively novel and in constant evolution. LangGraph benefits from being built on top of the already renowned LangChain#footnote("https://www.langchain.com/") ecosystem which adds to its reliability and ease of integration with other technologies.
Pydantic’s type safety will still be implemented within the project to enhance data validation and error handling.

The main responsibility of each agent is as follows:
- The *Intent Router* routes the user's query to the appropriate agents and accumulates query context.
- The *Clarify Query* clarifies the user's query if it is ambiguous or incomplete.
- The *Geocontext Retriever* retrieves the geospatial data relevant to the request.
- The *Guidelines Retriever* retrieves the relevant energy planning guidelines relevant to the query.
- The *Strategy Planner* plans the energy strategy based on the retrieved data and guidelines.
- The *Critic Answer* evaluates the proposed energy planning strategy and possibly restarts the whole process.
#highlight("TODO: plus en détails ?")

The various prompts are included in the appendix.
#highlight("TODO: mettre autre part cette info?")

With the overall solution defined, the following sections dig in the details of each agent and their implementation.

==== Intent Router <intent_router>

The intent router is a crucial component of the solution. It is the entrypoint of the system and orchestrates the different agents.

Upon receiving a user prompt, the agent analyzes the query to extract its underlying intent. This involves identifying these elements:
- The intent: specifies whether the query is "factual" (for e.g. requesting data) or "actionable" (seeking planning guidance, recommendations, or strategic advice).
- The location: the municipality name mentioned in the user request, if available.
- The aggregated query: a summary that combines all available context from the current conversation and the previous query into a single one.
- The conversation type: identifies the conversational context ; "new_analysis" (fresh query), "correction_request" (user questions the accuracy of a previous response) or 'follow_up' (user requests additional detail or expansion on the same topic).
- The need for clarification: defines whether more information is needed to understand what users wants (e.g., missing location, unclear intent, or vague request).
- The needs for memoization: specifies if the user provided explicit preferences, corrections to assumptions, or scope refinements that should be remembered for future queries (for e.g. the format used to summarize the retrieved data or only considering a single aspect from certain data points).

Implementation-wise, the output of the language model is constrained to a single Pydantic#footnote("https://docs.pydantic.dev/latest/") data schema.
While these models typically generate natural language responses, these complex multi-agent systems benefit from a structured output format that can easily be further processed.
This is possible thanks to OpenAI#footnote("https://openai.com/"), introducing the support for structured outputs in late 2024, a feature that has since been adopted by many providers.

The instructions #ref(<intent_router_prompt_system>) and #ref(<intent_router_prompt_user>) are designed to guide the language model on how to generate the correct output.
Few-shot prompting helps the language model by providing examples on how to complete the task, helping it generalize to subsequent prompts.
In this context, it facilitates the definition of the _aggregated_query_ and _needs_memoization_ fields.

All these fields except the aggregated query take value in a finite set of options (considering the _location_ field is either set or unset.).
Consequently, it is very easy to plan and orchestrate the following actions.

Since the application must offer a conversational experience, the previously determined state is updated and not overwritten.
Past fields are only updated if they differ from the new ones. As such, context and knowledge is properly accumulated over time.
Once the municipality is provided, for example, it is not needed anymore as the request is assumed to concern the same municipality.

On the other hand, when a request treats a different municipality, both the _context_tools_ and _context_constraints_ defined in #ref(<conversational_state>) are reset. This way, the data associated with the previously discussed municipality is cleared.
To ensure correct implementation, whenever a request concerns a different municipality, both the _context_tools_ and _context_constraints_ defined in #ref(<conversational_state>) are reset. This means the geospatial data and guidelines previously associated with the conversation are cleared.

On top of that, user-provided feedback and corrections shape the system's behavior allowing it to adapt to the user's preferences.
When there is a need for memoization, the system stores both the previous query (the _corrected_) and the current query (the _correctee_), in the database.

This highlights the interfaces of the intent router, as detailed in the #ref(<intent_router_design>) below:
#figure(image("figs/intent_router_design.svg", width: 80%), caption: "Intent router, interfaces")<intent_router_design>

#highlight("TODO: détail déf pydantic ??")
#highlight("TODO: parler que store parallèle ??")
#highlight("TODO: parler quelque part d'async??")
#highlight("TODO: parler au dessus coûte cher de faire un appel LLM")

An assumption still lies in the nature of the field _location_ as it is assumed to either be set or unset. A set location does not necessarily imply that it is a valid municipality, inscribed in the published Swiss official commune register#footnote("https://www.bfs.admin.ch/bfs/en/home/basics/swiss-official-commune-register.html").
A solution is proposed in the section #ref(<geocontext_retriever>, supplement: it => it.body).

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

Those two cases are both handled at once as a language model is prompted with the user's query and missing fields to generate and stream a response inquiring for further information or clarification (#ref(<ai_agent_design>), transition 3). The interfaces are illustrated in the #ref(<clarify_query_design>) below:
#figure(image("figs/clarify_query_design.svg", width: 80%), caption: "Clarify query, interfaces")<clarify_query_design>

In the following turn, the newly provided information is merged with the previously deduced intent as designed and presented in the section #ref(<intent_router>, supplement: it => it.body).

This task is supported by both #ref(<clarify_query_system_prompt>) and #ref(<clarify_query_user_prompt>).

With no _structural_ ambiguity left in the user's query, the intent router agent can now proceed to route the query to the geocontext retriever and guidelines retriever agents.

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
Mechanisms such as caching and geospatial indexing, a technique used in databases enabling quick identification of objects located within a particular geographical area, would be useful for greater scalability of the solution.

Datasets are often labeled as layers, as the data is organized according to the geospatial paradigm. Data is discretized into points, meshes, polygons and other spatial representations, all defined as a feature. Those features are independent geometries located in the space, without inherent relationships.
Thus, identifying relevant features within a municipality implies searching them inside its geographic boundaries, since no relation lies between these entities.

Although the GeoAdmin API enables searching for features in a given area, it is subject to a maximum number of 50 features retrieved per request.
Consequently, identifying them requires breaking down the search area into smaller sub-areas and querying each sub-area separately.

This has been implemented by first clipping settlements and centres of larger cities onto the municipality's geometry, optimizing the search area and applying a spatial tiling on top. Different layers obviously require different tiling sizes, depending on the number and resolution of features.

#set table(
  fill: (x, y) => { if calc.odd(y) and x != 0 { rgb("F7F9FA") } },
  stroke: (x, y) => {
    if y == 0 {
      (bottom: 0.7pt + black)
    }
    if x == 0 {
      (right: 0.7pt + black)
    }
    if y == 4 or y == 10 {
      (top: 0.3pt + black)
    }
  },
)
The #ref(<datasets_table>) presents the data sources incorporated in the solution:

#figure(
  rotate(-90deg, reflow: true, {
    show table.cell.where(x: 1): cell => {
      show regex("\b.+?\b"): it => it.text.codepoints().join(sym.zws)
      cell
    }
    table(
      columns: (3cm, 7cm, 8cm, 3cm, 3cm),
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

      [*Infrastructure*],
      [ch.bfe.statistik-wasserkraftanlagen],
      [Hydropower plants: statistics],
      [GWh/year],
      [Per plant],

      [], [ch.bfe.windenergieanlagen], [Wind energy plants], [GWh/year], [Per turbine],

      [], [ch.bfe.biogasanlagen], [Biogas plants], [kWh/year], [Per plant],
      [], [ch.bfe.kehrichtverbrennungsanlagen], [Waste incineration plants], [MWh/year], [Per plant],

      [], [ch.bfe.elektrizitaetsproduktionsanlagen], [Electricity production plants], [kW], [Per plant],

      [], [ch.bfe.thermische-netze], [Thermal networks], [MWh/year], [Per network],

      // [], [ch.swisstopo.swissboundaries3d-gemeinde-flaeche.fill], [Municipalities], [-], [Per municipality],

      // [], [ch.swisstopo.vec200-landcover], [Swiss land cover], [-], [Per surface],
    )
  }),
  caption: "Public datasets",
) <datasets_table>

#highlight("TODO: mieux décrire ce qu'il y a dans le tableau?")
#highlight("TODO: mettre les sources, liens vers datasets?")
#highlight("TODO: bouger label figure au dessus?")

The discretization of the different datasources showcases the importance of spatial tiling when dealing with this data.

In the average municipality, certain features are few and easily assessable whereas it is impossible to retrieve meaningful insights from the greater-resolution datasets (for e.g. the suitability of roofs, per roof pane) without additional processing or aggregation.

Therefore, the features within the municipality are (1) identified within the municipality, (2) aggregated to the municipality level and (3) brought back to the same GWh/year unit#footnote("Only applies to energy measures. Energy is deduced from power, in watts, assuming non-stop operation (24/7/365)."). This standardization allows for an easier and consistent comparison of scalars, on a yearly basis, crucial for interpretability but comes with drawbacks:
- The information of variability within the municipality itself is lost.
- The aggregation of data is a costly process, especially when doing this on the fly.

The first issue is partially recovered from the fact that the layers are displayed at their original resolution, in the web interface. This way, the variability can be easily visualized.

The second issue is mitigated depending on the nature of the data. The basic approach to aggregation requires the summation of values and is only needed for datasets that benefit from great precision. On the other hand, datasets that do not require such precision are subject to statistical estimation:
1. The spatial tiling is randomly sampled.
2. The features within the sampled tiling are identified and their values summed.
3. The sample mean (#ref(<sample_mean>)) and standard deviation (#ref(<std>)) are calculated.
4. The confidence interval is computed using a T-distribution and confidence level (#ref(<confidence_interval>)).
#set math.equation(numbering: "1.")

This random geographical sampling process is depicted in the #ref(<sampling_design>) below:
#figure(
  image("figs/tiling_design.svg", width: 100%),
  caption: "Random geographical sampling, municipality of Grône",
)<sampling_design>

Choosing the sampling size and confidence level is important for a proper statistical estimation. In this work, both parameters are set empirically and kept relatively large to benefit from lower computational costs, but without optimizing for the best possible accuracy.
As such, only the suitability of roofs and façades for use of solar energy is estimated using this technique as they are both datasets which showcase potential of exploitation rather than precise measurements and are well distributed in the geographic space.

#highlight("TODO: citer les chiffres confidence level...?")
#highlight("TODO: parler ajouter des modèles simu et autre dans les tools?")

With the data standardized and properly aggregated, the geocontext retriever agent must now be able to interact with it.

Previously, AI agents were described as autonomous systems able to operate and make decisions independently. These operations rely on tools.

A construction worker, for example, has different tools for different needs, such as a hammer for nails, a saw for cutting wood and a level for ensuring straight walls. The tools come with a set of instructions describing how to use them and what to expect from them.

The same applies to AI agents, which have, in this work, different tools allowing them to query, aggregate and retrieve data from the datasets in #ref(<datasets_table>).
Consequently, language models can leverage their natural language processing capabilities to choose the appropriate tools for the query.

An issue with the current approach is that when the _toolbox_ is too large, it becomes difficult for the language model to choose the right tools for the job.
This issue is addressed by exploiting the power of embeddings.

An embedding is a mathematical representation of data in a high-dimensional vector space where semantically similar information are mapped to nearby points.
This enables the system to embed the descriptions of the different tools and easily retrieve them semantically. On top of that, it is more efficient computation-wise than prompting the language model to choose them.
#highlight("TODO: définition terminologie? prompting")
#highlight("TODO: parler de reranking?")

When retrieving tools, the system computes the cosine similarity between both embeddings to quantify the semantic similarity (#ref(<cosine_sim>)).
Finally, the quartile coefficient of dispersion is measured against the distribution of retrieved scores (#ref(<qcd>)). This indicator provides a measure of the uniformity of the retrieved tools that is less sensitive to outliers than measures such as the coefficient of variation.
As such, uniform tools are provided to a language model, which is then prompted to choose the appropriate ones (#ref(<geocontext_retriever_system_prompt>)).

This approach reduces the overall computational cost while increasing the quality of tool selection.
#highlight("TODO: METTRE UNE AI NOTICE ET CHECKER VIM TEMP!!")

#highlight("TODO: faire un schéma détaillé du processus de l'agent?")

With the appropriate tools chosen, the system can effectively retrieve the data. It is simply added to the _context_tools_ field in the conversational state (#ref(<conversational_state>)), as presented in the #ref(<geocontext_retriever_design>) below:
#figure(
  image("figs/geocontext_retriever_design.svg", width: 80%),
  caption: "Geocontext retriever, interfaces",
)<geocontext_retriever_design>

Geospatial information is accumulated over the conversation turns, allowing for context-aware planning and consistent, spatially informed decisions. It is only reset when switching to a new municipality as it becomes invalid.

In the section #ref(<intent_router>, supplement: it => it.body), the validity of the location is not confirmed. This is directly implemented in the different tools above and routing of this agent (#ref(<ai_agent_design>)):
- If the location is non-valid, retrieving data raises an error and the request is routed to the clarify query agent (6).
- Otherwise, the query is sent to the strategy planner agent (7).

Once the relevant data is gathered, the next stage is for the strategy planner agent to analyze this information to conduct proper planning.

#highlight("TODO: citer external services database")
#highlight("TODO: dire requêtes API concurrent")
#highlight("TODO: parler spécifique map et système coordonnées?")
#highlight("TODO: ajouter tool energy needs, heuristique et détailler planification énergétique?")

==== Guidelines Retriever <guidelines_retriever>

The sole difference between enumerating the data, as collected in the geocontext retriever and proper energy planning lies in the measures that are taken in response to identified issues. Those measures are conditioned by guidelines, broken down into multiple sources.

The primary document called _Vision 2060 et objectifs 2035_ has been adopted in 2019 and sets intermediate targets for 2035 that take into account the energetical landscape of Valais/Wallis, current knowledge, as well as federal energy and climate policies with the ultimate goal of achieving a 100% renewable and indigenous energy supply in 2060.

Moreover, the _Plan directeur 2019_ adopted by the federal council on the 1st of May 2019, states the strategy for the canton's territorial development in the form of 49 information sheets, distributed across the five activity sectors: (1) _Agriculture, forest, landscape and nature_, (2) _Tourism and leisure_, (3) _Urbanization_, (4) _Mobility and transport infrastructure_ and (5) _Supply and other infrastructure_.

Finally, the legal framework is defined by two key legislative documents. Notably, the _RS 705.1 - Loi sur les constructions (LC)_ establishes the regulations for construction activities, while the _RS 730.1 - Loi sur l'énergie (LcEne)_ defines the objectives and requirements for sustainable energy supply.
#highlight("TODO: citer autrement rs -> bib?")
#highlight("TODO: citer autrement vision et plan directeur -> bib?")

These documents are specifically designed and structured to convey information to the public and come in a single #acr("PDF") and are available in both french and german. They are organized into sections, subsections or paragraphs which reference figures, tables, plots, past paragraphs and so on.

Visual structure does not necessarily imply a logical flow of information. A document can look and feel organized but still lack a proper machine readable structure.
In practice, it is neither realistic nor scalable to expect a human to manually extract all the key information needed for energy planning from such complex documents. Therefore, it becomes essential to delegate this task to the computer, enabling automated extraction and processing of documents.

When data lacks clear structure, it becomes difficult to extract information using algorithms or systematic procedures. However, advances in #acrpl("MLLM") offer a solution as these models are designed to process and understand information presented in various modalities such as text, images, audio and video. Paired with existing methods that are able to extract raw text from these documents, it has become easier to extract precise information from visually organized and heterogeneous documents by understanding not only the way information is displayed but also its underlying semantic meaning.

As such, a systematic approach is applied when extracting information from these documents:
- Raw text is extracted from the documents on a per-page basis.
- Each page is rendered into an image.
- Each rendered page and associated text are processed using MLLMs to retrieve key insights and interpret the information within the page.

This is supported by the #ref(<guidelines_retriever_data_extraction>).

The extracted information is formatted in markdown, utilizing headings to structure the summary. It is then broken down into smaller chunks, each chunk being a "chapter" derived from the markdown content. Since only individual chunks are considered in subsequent steps, there is no need to perform an analysis across neighboring pages to ensure that the information is retrieved from its full context.

Finally, the extracted information from each page is encoded into an embedding and stored in the local database, along with its associated chunks and metadata.

With clear guidelines extracted from documents in any format, it is necessary to identify those that are relevant to the user's query. Since those are already embedded, the related guidelines are simply those that are closest to the embedded request, as described in the #ref(<geocontext_retriever>, supplement: it => it.body) section is applied.
#highlight("TODO: revoir déf?")

An issue still lies in how the guidelines themselves are _designed_. While the objectives and figures outlined inside these documents concern the entire canton, this solution is only scoped to municipalities.
Therefore, these quantitative targets must be scaled down to reflect the municipality’s specific context and expectations.
#highlight("TODO: mettre un exemple?")

Identifying these key figures which need rescaling is not a simple task as broader context is required to assess that.
In order to achieve this, a language model is prompted with the task of identifying key figures that need rescaling (#ref(<guidelines_retriever_system_prompt>)).

Finally, they are multiplied by a factor corresponding to the ratio of the municipality's number of residents to the total population of the canton.
While this is a straightforward way to scale targets, proper rescaling should take into account the economic activity, energy landscape and industrial presence in the municipality to ensure a more accurate adjustment.

The adjusted guidelines are accumulated onto the _context_constraints_ field in the conversational state (#ref(<conversational_state>)).
Similarly to the geospatial information described in the #ref(<geocontext_retriever>, supplement: it => it.body) section, the processed guidelines are accumulated in the state as the conversation goes on and only cleared when switching to a new municipality.

These interfaces are shown in the #ref(<guidelines_retriever_design>) below:
#figure(
  image("figs/guidelines_retriever_design.svg", width: 80%),
  caption: "Guidelines retriever, interfaces",
)<guidelines_retriever_design>

With the relevant guidelines retrieved and rescaled, the query is routed to the strategy planner agent (#ref(<ai_agent_design>), transition 7) which will use them as clear constraints.

#highlight("TODO: mettre un premier chapitre preprocessing?")
#highlight("TODO: cite plus tard que Retrieve multiple chunks recreate the context")
#highlight("TODO: reformuler et enlever les formulations 'we' ?")
#highlight("TODO: dire que je traduis en anglais ou pas, pff")
#highlight("TODO: citer modèle utilisé MLLM et embeddings ?")
#highlight("TODO: citer comment traiter données communales")
#highlight("TODO: inclure prompts dans bibliographie")

==== Municipal citizen profile

Before delving into the strategy planner agent, it is necessary to assess how the solution could be enhanced by incorporating additional municipal data.
This section is deliberately included in the #ref(<system_design>, supplement: it => it.body) chapter, as this aspect of the solution was envisioned from the start and a proof of concept has been developed.

Switzerland has three levels of political authority: the Confederation (federal government), the cantons (states) and the communes (municipalities).

Municipalities have an administrative and regulatory role in different over residents, especially regarding housing and energy use.
Consequently, municipalities gather extensive data that can be highly valuable to establish proper energy planning strategies, at a local level.

When residents apply for building permits, they are required to submit detailed information regarding the construction.
Energy efficiency standards enforce mandatory requirements, which are evaluated via an energy profile of the construction.

These applications become part of the citizen's record and provide relevant information and contain key details about the type of heating system, the annual energy consumption and the energy reference surface of the construction.

On top of that, residents who install solar photovoltaic systems and solar thermal systems apply for subsidies, provided through various programs.
Detailed technical specifications of the installation are required, including its power.

After being assessed and approved by the municipality, these applications are added to the official record.

Moreover, municipalities are actively working to digitalize these processes.
Currently, most documents are scanned and stored in a digital format (typically PDF).

As described in the #ref(<guidelines_retriever>, supplement: it => it.body) section, MLLMs facilitate the extraction of information from these documents.
The only difference lies in how those models are leveraged.

Instead of inquiring the model to summarize or retrieve key insight, a simple citizen profile that is valuable for energy planning is defined#footnote("This profile was defined with the help of the supervisors and could easily be extended.") and searched for, inside each page of the citizen's record (#ref(<municipal_citizen_profile>)):
- The parcel number
- The energy reference surface of the construction, in square meters
- The type of heating system
- The annual energy consumption, in kilowatt-hours per annum
- The power of the solar photovoltaic system, in kilowatt-peak

In the end, the individual pages results are aggregated into a single resident energy profile.

This procedure was tested against a typical citizen record, provided by Prof. Jessen Page. The #ref(<citizen_profile_example>) below provides the anonymized (without _parcel_number_), resulting profile:

#figure(
  code()[
    #raw(
      "{
        'parcel_number': xxx,
        'sre': 169,
        'consumption_heating': 185.3,
        'source_heating': 'PAC air/eau',
        'power_pv': 9.68
}",
      lang: "json",
    )
  ],
  caption: "Citizen profile, example",
) <citizen_profile_example>

In a production scenario, these profiles would be generated automatically by setting up data pipelines and stored in a database for further use.

This implementation confirms the feasibility and systematic approach to extracting information from municipal records, supporting the broader vision of the solution's ability to leverage citizen data and gain further insight from these non-public sources.

With this proof of concept demonstrated, the next step covers the design and implementation of the strategy planner agent.

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

The state interfaces are presented in the #ref(<strategy_planner_design>) below:
#figure(
  image("figs/strategy_planner_design.svg", width: 80%),
  caption: "Strategy planner, interfaces",
)<strategy_planner_design>

The factual queries are treated by both #ref(<generate_answer_factual_system_prompt>) and #ref(<generate_answer_factual_user_prompt>) whereas actionable queries are handled by #ref(<generate_answer_actionable_system_prompt>) and #ref(<generate_answer_actionable_user_prompt>).
The conversational context is simply broken down and included in the prompts.

The language model response, which is the answer to the user's query, is streamed to the web interface (#ref(<ai_agent_design>), transition 8).

While the user examines the response, it is sent to critic agent (transition 9), which will evaluate its quality and act accordingly.

==== Critic <critic>

One of the main challenges in designing a conversational AI solution for real-world problems is ensuring the accuracy and relevancy of the response.
Decomposing the complex task of energy planning into smaller steps, each addressed by a dedicated agent, allows for specialized solutions, leading to more precise and context-aware interactions.

Inaccurate responses issue from different factors such as incorrect data or lack of context.
In this work, an element that may be even more impactful is the interpretation of this same context, curated by the different agents involved in the workflow.
These interpretations errors typically include:
- Mathematical errors where data points are added or subtracted to support insights.
- Flawed conclusions from different metrics incorrectly treated or incorrect assumptions.

As such, a language model is prompted the response generated in the <strategy_planner> with the data points and guidelines that shaped it (#ref(<critic_answer_system_prompt>)).
On top of that, the number of residents in the municipality and its exploitable area are both included, providing extra context that helps the model assess the feasibility of the proposed strategy.

#figure(
  image("figs/critic_design.svg", width: 80%),
  caption: "Critic design, interfaces",
)<critic_design>
Its output is a boolean value (#ref(<critic_design>), _retry_) that indicates whether the response has been interpreted correctly based on the rules above.

If it the response is not satisfactory, the complete process is restarted as if the user had just prompted the system (#ref(<ai_agent_design>), transition 10).
A maximum of three attempts are allowed before the workflow is not restarted anymore.

At this point, the user's request has been answered and the system is ready to receive a new request, refining the proposed energy planning.

This concludes the design and implementation of the AI agent responsible for energy planning.
#highlight("TODO: conclusion plus longue jsp")

=== Web Interface

Designing and providing an interface that is user-friendly and convenient for the user to interact with the AI system is key to the adoption of the product.
The exactitude and accuracy of the AI system weighs heavily on the user's satisfaction but so does the user experience and presentation that is offered.

In recent times, the tendency shifted from traditional desktop applications to web-based interfaces, offering greater accessibility for devices such as smartphones.
Therefore, a web interface was developed to provide a seamless experience for users.

On the development side, the interface was implemented using React#footnote("https://react.dev/"), a popular framework released by Facebook (now Meta) in 2013. React offers a declarative and efficient way to build user interfaces, offering a clean and modular approach using components.

In reality, the choice of framework is not particularly critical in this context. Dozens of frameworks claim to revolutionize the way developers build web applications, but all lead to similar outcomes despite different approaches and philosophies. Past experiences with React and its ecosystem made it a comfortable and efficient choice for this project.

#highlight("TODO: parler vite, bun et docker???")

The primary feature of the web interface is the ability to send a prompt to the AI system and have the response streamed back, in real time.
This is achieved using #acr("SSE"), a one-way communication protocol where the server (here the AI agent) pushes events to the client.
In addition to streaming tokens as events, continuous status updates are emitted from the agents to provide live feedback about the progress of the AI solution.

The application being event-driven, SSE was chosen over classic polling mechanisms. Polling requires repeated requests from the client to the server, which can increase both server load and response latency whereas SSE maintains a single persistent connection on which events are pushed.

On top of that, the use-case of energy planning for municipalities greatly benefits from a map, displaying the assessed data points.
This functionality recovers a problem covered in the #ref(<geocontext_retriever>, supplement: it => it.body) section: the loss of information due to aggregation.

All datasets referenced in #ref(<datasets_table>) provide layers, along with their discretized features. Interpreting them visually preserves the local variation and allows for a more nuanced understanding of the reported aggregated values.

These data points and their sources are also presented in the interface, allowing users to assess and verify the accuracy of the reported energy planning. This transparency is offered to further build user trust in the solution.

On the implementation side, OpenLayers#footnote("https://openlayers.org/") is a robust and flexible open-source library for building interactive web maps.
Native support for #acr("WMTS") enables the integration of tiled map services such as those provided by GeoAdmin. These tiles are high-resolution images enabling efficient and scalable map rendering by only loading visible portions of the map. GeoAdmin notably makes use of OpenLayers in their services.

With the ultimate goal of assisting users i energy planning, downplaying the importance of durability aspects of the solution would be a significant oversight.
AI progress is often celebrated, yet the energetical impact of these systems is frequently overlooked.
While a complete evaluation of the energetical footprint of the solution is clearly beyond this work's scope, a simple approach has been implemented to sensitize users to the matter.
#highlight("TODO: en faire un sous-chapitre?")

For each user prompt, the cumulated number of tokens that are input and output from the agents in #ref(<ai_agent_design>) is recorded.

The cumulative count of prompts and average token count per prompt are incrementally calculated and stored in the database.

Along that, Welford's online algorithm allows for the calculation of the variance, in a single pass.
It defines a recurrence relation for updating the sum of squared differences from the current mean, allowing to compute the variance incrementally.
#highlight("TODO: référencer wikipedia correctement")
#highlight("TODO: inclure math ahahahha")

This algorithm is numerically stable and does not require storing all the data points, reducing the memory footprint of the system.

To sum up, the token utilization of users is tracked in the form of three metrics only: (1) the average token count per prompt, (2) the sum of squared differences from the current mean, from which the variance can be computed and (3) the cumulative count of prompts.

When users prompt the AI, the cumulative token count for that run is monitored. This value is then compared against the user's sampled token utilization distribution using a z-score to measure how far the new usage deviates from the user's average.
Accordingly, the token usage of the current prompt is categorized into one of the predefined categories: "bad", "average", or "good" ; each associated with a color pelet displayed in the interface.
#highlight("TODO: mieux définir zscore??")
#highlight("TODO: mettre les maths?")

Moreover, an energy consumption analogy is presented alongside the pelet.
A simple heuristic of 3 Joules per token is used to estimate the energy consumption of the current prompt.
It is then expressed as the equivalent duration, in minutes, a 10 W LED light bulb could run with the same amount of energy.

#highlight("TODO: CITER LE PAPIER ET LE METTRE DANS LA BIBLIOGRAPHIE")
#highlight("TODO: mettre les maths?")

This implementation is strictly meant to raise awareness about the energy consumption associated with AI usage.
In this solution, requesting factual data from the system is more energy-efficient than inquiring actionable planning.
This is because factual queries involve fewer agents in the workflow defined in #ref(<ai_agent_design>).

#highlight("TOOD: concept de persistance???")
#highlight("TODO: mettre des screenshots?")
#highlight("TODO: mettre un schéma?")

With that, the implementation details of the web interface are clarified. This highlights its role as being on par with the AI agent solution itself.

== Limitations <limitations>

#highlight("TODO: en faire un chapitre par composant au dessus ou en dehors de la partie méthodologie?")
#highlight("TODO: checker orthographe de tout le rapport!!")

Every technical solution, regardless of how well designed and implemented, is subject to different limitations.
These constraints often stem from underlying assumptions made during the design process or specific implementation details.

Identifying them is a first step towards a self-evaluation of the solution. Please note that the limitations outlined in this section are not exhaustive.

When integrating external services and data sources, it is very difficult to assess the quality and exactitude of the data.
Inaccuracies lead to incorrect interpretations and conclusions reported by the system, degrading the performance and reliability of the system.

The various offices referenced in the #ref(<geocontext_retriever>, supplement: it => it.body) section, commissioned to retrieve and publish data, address these issues by implementing different quality assessment procedures.

The SFOE, for example, only tolerates a small proportion of errors and gaps in the data they collect#footnote("https://www.bfs.admin.ch/bfs/de/home/register/personenregister/registerharmonisierung/qualitaet-datenlieferung.html").

During the development of the solution, one such observation was made.
When assessing the energy production of waste incineration plants, both the electricity and heat production are reported.
In the municipality of Sion (#ref(<datasets_table>)), the values that are reported for the waste treatment plant of central Valais/Wallis (UTO, now Enevi#footnote("https://www.enevi.ch/fr")) are those of the year 2017.

These outdated values surely lead to inexact deductions when establishing the profile and strategy of a municipality.
Establishing long-term roadmaps and strategies based on outdated data can surely lead to inaccurate predictions and ineffective planning.

It is virtually impossible to estimate the full extent of inexactitudes and underlying issues within the data. Therefore, the sole solution is to place trust in these organizations.

Besides that, the interpretation that is made of the data is as important as its quality. Poor understanding may promote false biases and false deductions.

In this work, the solution was organized around agents, each responsible for specific tasks. As such, smaller reasoning language models were leveraged to reduce the computational cost of the system.
However, a bigger, more capable model was used in the strategy planner agent.

Although not quantified in this context, larger models are generally better at processing complex accumulated context and providing more coherent interpretations.
Recent research highlights that larger language models demonstrate superior emergent reasoning and contextual integration abilities.
This is shown in the paper _Emergent Abilities of Large Language Models_, published in the _Transactions of Machine Learning Research_, in 2022.
These capabilities are relevant for the strategy planner agent which must synthesize large context to deduce energy planning insights and recommendations.
#highlight("TODO: citer proprement")

To support this, greater models run on _Calypso_, a sandbox infrastructure designed and reserved for students of the bachelor program.
Nevertheless, the largest model that fits on this infrastructure (around 8 billion parameters) remains relatively small compared to state-of-the-art large language models (>150 billion parameters).
Exploring the use of -_really_- large language models in future work could yield improvements in interpretation and overall quality of the system.
#highlight("TODO: mettre un exemple prompt mixin résutlats??")

Furthermore, a strong limitation of the solution is the inability of the user to provide and induce bias in the system#footnote("While no extensive jailbreaking or bias testing was conducted, all attempts made during the development process to introduce bias into the system were unsuccessful.").

This might be seen as a good thing, as it guarantees consistency in the output but this also means that the system is less flexible and adaptable to the individual preferences, explained in the #ref(<intent_router>, supplement: it => it.body) section.
While this ensures consistent output, it also reduces the flexibility and adaptability of the system to put into service all user preferences.

These instructions are distinguished into two categories: presentation directives (1) and custom data instructions (2).

Presentation directives define how the data should be formatted (table, bullet points...) and which aspects of the data should be prioritized or highlighted. This structures the response and emphasizes specific details that are key to the users.

Custom data instructions, on the other hand, attempt to substitute the official data sources retrieved by the system with user-provided knowledge.
Informed users may provide more accurate and precise figures regarding the datasets referenced in #ref(<datasets_table>) but their contributions are completely ignored as the solution prioritizes its own retrieved data.
#highlight("TODO: mettre un exemple de bias data qui marche pas")

The system is guided by the different prompts to strictly operate based on the workflow defined in #ref(<ai_agent_design>) and offer assistance in energy planning tasks.
Providing more detailed instructions in the response generation prompt may enable this behavior.
#highlight("TODO: parler qqepart de prompt engineering?")

Finally, a stronger limitation is induced by the choice of agentic paradigm.
Lately, coding agents, AI agents capable of autonomously generating and executing code, have gained popularity.

Leveraging them broadens the scope of tasks that can be handled by the agents. General use cases for these agents include data processing and analysis, interaction with APIs and arithmetics.
The retrieveal, processing and aggregation of the data sources presented in the #ref(<geocontext_retriever>, supplement: it => it.body) section would be an excellent application of their capabilities.

The classic agent architecture adopted in this work ensures a more transparent and traceable workflow where the role and interface of each agent is clearly defined.
While this design choice facilitates maintainability and allows for a more reliable and structured handling of complex geospatial data, it would still be interesting to explore the paradigm shift towards coding agents.

Outside of the data retrieval and processing, these agents would greatly enhance the critic of the system's response (#ref(<critic>, supplement: it => it.body)).
A typical energy profile, including relevant indicators from the datasets in #ref(<datasets_table>) could be pre-defined and, for example, standardized to a per-resident basis.

By doing so, the agent could evaluate the accuracy of the values presented in the energy planning report, converting them to the same reference scale and comparing them against the average profile.
This would enable the identification of subtle inconsistencies, leading to a more robust and reliable implementation.

In summary, these points illustrate some of the current limitations inherent to the system. The following chapter presents the results obtained from the implemented solution.

= Results <results>

This chapter presents the empirical findings from the assessment of the implemented system.
A structured testing methodology is established, facilitating the evaluation of the solution's primary objective: assisting users in municipal energy planning.

== Evaluation Methodology

To begin, it is essential to establish the methodology and criteria used to evaluate the system's performance.

There is no definitive ground truth in energy planning, making it difficult to quantify the accuracy of the reported recommendations and strategies.
Consequently, qualitative observations provide valuable insight into the practical effectiveness of the solution.

These insights are provided by two sources:
- An expert assessment, provided by Prof. Jessen Page
- An automated evaluation using a LLM-as-a-judge benchmarking framework

The latter leverages language models to mimic expert judgment, assessing the response against a set of predefined criteria.
This approach provides a scalable and consistent alternative to human evaluation.

=== LLM-as-a-judge Benchmarking Framework
#highlight("TODO: vérif abus de langage g-eval dans discussions??")
Unlike human experts which may emphasize different aspects depending on their interpretation or even mood, language model evaluation is driven by clear rules, ensuring consistency and uniformity across all cases. The G-Eval metric, as introduced in the #ref(<state_of_art>, supplement: it => it.body) section, summarizes the score of the criteria and quantifies the quality of the response.

The criteria for the LLM-as-a-judge benchmarking framework are defined as follows:
1. Data interpretation: assesses whether the response uses only relevant data, maintains mathematical accuracy, distinguishes energy types, preserves units and handles zero values correctly.
2. Methodology alignment:
  - For factual queries: checks clear data analysis, insight identification and avoidance of proper plannification.
  - For actionable queries: evaluates structured planning, guidelines integration and expert positioning.
3. Municipal relevance: rates feasibility at local scale, direct query alignment, consideration of local context and actionable next steps.
4. Technical compliance: checks language consistency, correct structure, citation format and completeness of required sections.

And evaluated according to the following scale, presented in #ref(<scoring_grid_llm>):
#set table(
  fill: (x, y) => { if calc.odd(y) { rgb("F7F9FA") } },
  align: (x, _) => if x == 0 { center } else { left },
  stroke: (x, y) => {
    if y == 0 {
      (bottom: 0.7pt + black)
    }
  },
)
#figure(
  table(
    columns: (auto, 80%),
    table.header([Score], [Description]),
    [1], [Fundamental misunderstanding, major errors, or advice that is actively misleading or harmful.],
    [2], [Wrong methodology, significant mistakes, or advice that is not actionable or fails basic requirements.],
    [3], [Basic competence, but significant limitations or generic advice.],
    [4], [Good quality, minor issues, genuinely useful for municipal planning.],
    [5], [Exceptional—accurate, specific, actionable, perfect methodology alignment.],
  ),
  caption: "Ordinal evaluation grid for criteria in the benchmarking framework",
) <scoring_grid_llm>

These criteria are based on the most frequent types of errors, encountered during the weekly assessment of the solution.

These instructions are summarized and included in the #ref(<benchmark_system_prompt>). The accumulated context is included, helping the model to evaluate the accuracy of the generated response.

Finally, the individual benchmark scores are aggregated into a single score by calculating the arithmetic mean and then rescaled linearly to the [0,1] interval.

=== Expert Assessment Framework

The former human insight is naturally shaped by the domain knowledge and nuanced judgment of the expert, leading to a more informed assessment.

While the automated evaluation is constrained to a rigid scoring grid, the expert evaluation is richer as observations extend beyond these predefined criteria and reflects interpreted, context-specific priorities.
As such, it is not feasible to enforce strict scoring rules to the expert.

However, the grid presented in #ref(<scoring_grid_expert>) acts as a reference point and guides the nuanced and _unlimited_ qualitative feedback to a single quantitative score, allowing for further comparison and analysis:
#figure(
  table(
    columns: 3,
    table.header([Score], [Label], [Description]),
    [1],
    [Not relevant],
    [Information is completely off-topic ; does not answer the question, is generic or incoherent in the context of energy planning.],

    [2],
    [Weakly relevant],
    [Some elements are related to the subject ; the response is mostly vague, imprecise, or off-topic. It could mislead an expert.],

    [3],
    [Moderately relevant],
    [Response is generally on theme but remains partial, imprecise, or incomplete ; requires significant corrections to be useful.],

    [4],
    [Relevant],
    [Information is correct, targeted and generally adapted to the question ; minor adjustments may be needed but it is usable for planning.],

    [5],
    [Highly relevant],
    [Information is perfectly aligned with the question, complete and contextualized ; no corrections are necessary and it is ready to be used as-is for decision-making.],
  ),
  caption: "Ordinal evaluation grid for the expert",
) <scoring_grid_expert>

It is important to note that the G-eval and expert scores cannot be interpreted and directly compared, each being grounded in a distinct evaluation framework.

G-eval offers a standardized framework with set evaluation criteria, allowing for a more consistent and reliable benchmarking. This enables the comparison of different solutions under identical conditions.

As G-eval relies on LLMs, scores may showcase some degree of randomness. This is mitigated by multiple runs and aggregated scores, offering a stable estimate.

By presenting both evaluation methods, the objectivity of an automated scoring is complemented by the more practice-oriented expert judgment
This dual approach treats both methodological rigor and contextual relevance to assess the quality of the solution.

For consistency, expert scores are also rescaled linearly to the [0, 1] interval.

Both frameworks are inherently ordinal with each value being _qualitatively_ described.
However, the rescaled scores are treated as continuous variables for statistical purposes, enabling analytical comparisons.

#pagebreak()
=== Test Dataset

With the evaluation frameworks introduced, the next step is to define the test dataset.
This dataset consists of nine prompts and establishes the basis for assessing the performance of the solution:
#set table(
  fill: (x, y) => if calc.odd(y) { rgb("F7F9FA") },
  align: (x, _) => if x == 0 { center } else { left },
)
#figure(
  table(
    columns: 2,
    table.header([No°], [Prompt]),
    [1], [What is the current energy consumption per energy vector and per consumer type in Sion?],
    [2], [How is this demand expected to evolve until 2050?],

    [3], [What energy efficiency measures should be considered to reduce this consumption?],
    [4], [Can you tell me the amount of CO2 associated to this demand?],

    [5], [What are potential sources of renewable energy in Sion (GWh/an for each source)?],
    [6], [How much of this potential is currently exploited?],

    [7], [How much is expected to be exploited in the future?],
    [8],
    [Can you provide me with a map of the electricity grid and potential PV production on roofs and other surfaces?],

    [9], [Can you provide me with a map of heat/cold demand density and potential sources of heat/cold?],
  ),
  caption: "Test dataset prompts for municipal energy planning in Sion",
) <test_dataset>

The dataset is aligned within the specific scope of available data sources, inherently restricting its size.
It is crafted by Prof. Jessen Page to support energy planning for the municipality of Sion.

With the groundwork established, the following tables present the results that will be discussed in the next section.
Both the expert assessment and benchmarking are conducted using the version of the solution as of June 20, 2025#footnote[Code at commit `c2bea64` in the repository.].

== Evaluation Results

Two different configurations of the solution, each using different language models, are benchmarked.

A smaller configuration is defined for further comparison against the baseline.
This baseline, a _larger_ configuration, leverages bigger language models for the strategy planner agent as introduced in the #ref(<limitations>, supplement: it => it.body) section.

The #ref(<configurations>) below breaks down their composition:

#set table(
  stroke: (x, y) => {
    if y == 2 {
      (bottom: 0.7pt + black)
    }
    if x > 1 {
      (left: 0.3pt + black)
    }
    if x == 0 {
      (right: 0.3pt + black)
    }
    if x == 1 and y == 0 {
      (top: 0.3pt + black)
      (bottom: 0.3pt + black)
    }
  },
  align: (x, y) => {
    if (x == 0 or x == 5) and y == 2 { (bottom) } else { auto }
  },
)
#figure(
  table(
    columns: (auto, 2.5cm, 2.5cm, 2.5cm, 2.5cm, auto),
    table.header(
      table.cell(rowspan: 2, []),
      table.cell(colspan: 4, rowspan: 2, [Agent language model]),
      table.cell(rowspan: 2, []),
      [Configuration],
      [*Intent Router*],
      [*Geocontext Retriever*],
      [*Guidelines Retriever*],
      [*Strategy planner*],
      [*G-eval Evaluator*],
    ),
    [*Small*], [qwen3:1.7b], [qwen3:1.7b], [qwen3:1.7b], [qwen3:1.7b], [deepseek-r1:8b],
    [*Large*], [qwen3:1.7b], [qwen3:1.7b], [qwen3:1.7b], [*qwen3:8b*], [deepseek-r1:8b],
  ),
  caption: "Agent language model by configuration",
) <configurations>

Qwen is a large language model family built by Alibaba Cloud#footnote("https://www.alibabacloud.com/en?_p_lc=7"). It offers tool-using abilities, reasoning and model size, making it the only currently available option on Ollama that is viable for running lightweight, yet capable agents.

Deepseek models, on the other hand, are developed by Deepseek, a company funded by the High-Flyer#footnote("https://www.high-flyer.cn/en/fund/") hedge fund and whose models support similar capabilities to Qwen.
The R1 model (8B), a strong general-purpose language model, is used as the evaluator across both configurations to clearly distinguish the model family from that of the agent components, reducing potential bias in the assessment.

While the expert assessment and G-eval benchmarking are not comparable _as is_, the #ref(<comparison_test_human_llm>) presents their scores across the nine prompts, side by side, and offers an interesting perspective:

#set table(
  stroke: (x, y) => {
    if y == 2 {
      (bottom: 0.7pt + black)
    }
    if x > 0 and y > 0 {
      (left: 0.3pt + black)
    }
    if x == 2 and y == 0 {
      0.3pt + black
    }
    if x == 3 and y == 2 {
      (right: 0.3pt + black)
    }
  },
  align: (x, _) => if x == 0 { right } else { center },
)
#figure(
  table(
    columns: 4,
    table.header(
      table.cell(rowspan: 2, []),
      table.cell(rowspan: 2, []),
      table.cell(rowspan: 2, colspan: 2, [G-eval (mean ± st.d.), 10 runs]),
    ),
    [Prompt No°], [*Expert score*], [*Large configuration*], [Small configuration],
    [1], [0.50], [0.39 ± 0.15], [0.23 ± 0.08],
    [2], [0.25], [0.43 ± 0.11], [0.26 ± 0.10],
    [3], [0.50], [0.44 ± 0.11], [0.36 ± 0.06],
    [4], [0.25], [0.37 ± 0.22], [0.23 ± 0.08],
    [5], [0.75], [0.39 ± 0.17], [0.38 ± 0.00],
    [6], [0.50], [0.41 ± 0.15], [0.23 ± 0.08],
    [7], [0.75], [0.44 ± 0.10], [0.28 ± 0.10],
    [8], [0.25], [0.49 ± 0.13], [0.38 ± 0.00],
    [9], [0.25], [0.50 ± 0.11], [0.21 ± 0.06],
  ),
  caption: "Expert assessment and G-eval scores on test dataset, per prompt.",
) <comparison_test_human_llm>

#highlight("TODO: inclure résultats testcase dans appendix??")

Finally, the scores per benchmarking criteria and per query _intent_ type (#ref(<conversational_state>), either factual or actionable), for both configurations, are depicted in the #ref(<comparison_criteria>) below:
#set table(
  stroke: (x, y) => {
    if y == 2 {
      (bottom: 0.7pt + black)
    }
    if x > 0 and y > 0 {
      (left: 0.3pt + black)
    }
    if x == 1 and y == 0 {
      0.3pt + black
    }
    if x == 2 and y == 2 {
      (right: 0.3pt + black)
    }
    if (x == 1 or x == 3) and y == 3 {
      (right: 0.3pt + black)
      (bottom: 0.3pt + black)
    }
    if x == 0 and y == 4 {
      (top: 0.3pt + black)
    }
  },
  fill: (x, y) => {
    if y < 4 and x == 0 { none } else {
      if calc.odd(y) { rgb("F7F9FA") }
    }
  },
  align: (x, _) => if x == 0 { left } else { center },
)
#figure(
  table(
    columns: 5,
    table.header(
      table.cell(rowspan: 3, []),
      table.cell(rowspan: 3, colspan: 4, [Score (mean ± st.d.), 10 runs]),
    ),
    [], table.cell(colspan: 2, [Factual]), table.cell(colspan: 2, [Actionable]),
    [*Criteria*], [Large], [Small], [Large], [Small],
    [Data interpretation], [0.39 ± 0.21], [0.11 ± 0.13], [0.44 ± 0.25], [0.18 ± 0.12],
    [Guideline application],
    [0.30 ± 0.35],
    [0.14 ± 0.13],
    [0.52 ± 0.31],
    [0.30 ± 0.21],
    [Municipal relevance],
    [0.63 ± 0.21],
    [0.50 ± 0.00],
    [0.48 ± 0.30],
    [0.39 ± 0.26],
    [Source citations], [0.20 ± 0.30], [0.15 ± 0.31], [0.34 ± 0.35], [0.33 ± 0.38],
  ),
  caption: "LLM-as-a-judge benchmark score on test dataset, per criteria and per query intent.",
) <comparison_criteria>
#highlight("TODO: en faire un plot??")

It is important to note that the distribution of prompts by intent is unbalanced as only two prompts from the test dataset (#ref(<test_dataset>)) are classified as "factual", while the seven remaining ones are classified as "actionable".

These tables summarize and introduce further comparison between the expert evaluation and G-eval benchmarking framework, assessing differences between the baseline configuration and a smaller one.
With the different results presented, the following chapter discusses their implications.
#highlight("TODO: changer mot tables en plots??")

= Discussion

Before analyzing the implications of the results, it is important to restate the research question: evaluating the effectiveness and reliability of AI agents for urban energy planning, along with an analysis of their strengths and weaknesses.

This chapter addresses this objective by interpreting the results and connecting them to the research question.

== Evaluation Paradigms

The #ref(<results>, supplement: it => it.body) chapter differentiates the two evaluation frameworks: (1) the expert assessment framework and (2) the G-eval benchmarking framework.

As both rely on different methods and criteria, they provide complementary insights rather than direct comparability.
The expert evaluation provides nuanced and domain-informed feedback while the G-eval framework delivers a standardized and consistent assessment.

The #ref(<comparison_test_human_llm>) presents both frameworks and their scores, side-by-side, against each prompt of the test dataset.

The relationship between these two frameworks is assessed by leveraging Spearman's rank correlation coefficient (#ref(<spearman>).
This coefficient measures the monotonic relationship between the rankings of two variables, indicating how well the order of one variable matches the order of the other. Like other correlation coefficients, it varies between -1 (perfect inverse correlation) and +1 (perfect correlation), with 0 implying no correlation at all.
#highlight("TOOD: citer spearman?")

This coefficient strictly applies to ordinal data. As both score distributions originate from a ranking, the coefficient is applicable to compare the relative ranking of prompts of the two frameworks.

As such, the calculated Spearman correlation coefficient between the expert evaluation and the G-eval framework is -0.23 for the larger configuration and 0.36 for the smaller configuration, indicating no meaningful correlation between the two frameworks.
This divergence demonstrates that the frameworks capture distinct dimensions than those of the expert evaluation, validating the complementary nature of the evaluation methods.

Although the smaller configuration shows a weak trend, it is not statistically significant as the sample size is too small. Consequently, the G-eval framework is, at most, a complementary tool to the expert evaluation.

== Performance Analysis

On top of that, the same #ref(<comparison_test_human_llm>) depicts the average G-eval scores for both configurations, per prompt. Visually, the larger configuration consistently outperforms the smaller configuration.

This can be formally assessed by conducting a Wilcoxon signed-rank test, whose null hypothesis evaluates if two related samples (paired samples) come from the same distribution.
Besides that, it does not assume an underlying normal distribution, making it more robust to outliers.
#highlight("TODO: citer wilcoxon")

Therefore, it is applied to the paired scores of the larger and smaller configurations with the one-sided alternative hypothesis that the larger configuration significantly outperforms the smaller one.
The resulting p-value is 0.009 ($equiv$ 0.9%), indicating a statistically significant difference between the two configurations considering a 5% confidence level.

The null hypothesis is hence rejected and confirms the alternative hypothesis: the larger configuration, indeed, outperforms the smaller configuration in a per-prompt basis. This confirms the initial expectation that larger language models, incorporated into the AI agent solution, yield better results.

Analyzing the per-criteria scores across for each query intents (#ref(<comparison_criteria>)) provides a more detailed view of how the two configurations scores across specific queries and fields.

// expliquer pq telle différence dans les résultats
// pros and cons
// PAS ASSEZ DE SAMPLES POUR COMPARER SI EXPERT NOTE MIEUX SUR LES ACTIONABLE MAIS ASSESS INDICATEUR SUR G-EVAL
// volatilité larger ??
// est-ce que c'est la faute de g-eval et de ces critères ?
// discuter qu'est-ce qui est faux quand expert met un mauvais score
// EVALUER LES OBJECTIFS DU TRAVAIL
// 9. *Discussion*: Interprets the results, discusses implications, and relates findings to the research question.
// 10. *Conclusion*: Summarizes the main findings, contributions, and suggests future work.

= Conclusion

// amélioration graphe
// MCP
// train classificateur guidelines, fine tune, ...
// train classificateur intent
// fuzzy search <- amélioration plutôt que limitation et dire que utiliser un dictionnaire difficile car mises à jour fréquentes par exemple ici ou plus bas ??
// support mobile

// PROS AND CONS DU PROJET, LLM POUR LA TACHE
// importance prompt engineering?
// EVALUER LES OBJECTIFS DU TRAVAIL ICI?


#pagebreak()
#heavy-title(i18n(doc_language, "bibliography-title"), mult: 1, top: 0.5em, bottom: 0.3em)
// generate bib file RIS script (https://www.bruot.org/ris2bib/)
#bibliography("bibliography.bib", full: true, style: "ieee", title: none)
#highlight("TODO: reexport bookmarks")
#highlight("TODO: mettre une deuxième bib. technique uniquement!!")
// #bibliography(("bibliography.bib", "technical_reference.bib"), full: true, style: "ieee", title: none)

//////////////
// Appendices
//////////////
#cleardoublepage()
#appendix-page()
#pagebreak()

#heavy-title("Prompts", top: 1em, bottom: 1em)

#let prompt_creation = read("code/prompt_creation.txt")
#figure(
  code()[
    #raw(prompt_creation)
  ],
  caption: "Prompt engineering prompt",
  kind: "prompt",
  supplement: [Prompt],
) <prompt_creation>

#let intent_router_prompts_system = read("code/intent_router_prompt_system.py")
#figure(
  code()[
    #raw(intent_router_prompts_system, lang: "python")
  ],
  caption: "Intent router, system prompt",
  kind: "prompt",
  supplement: [Prompt],
) <intent_router_prompt_system>

#let intent_router_prompts_user = read("code/intent_router_prompt_user.py")
#figure(
  code()[
    #raw(intent_router_prompts_user, lang: "python")
  ],
  caption: "Intent router, user prompt",
  kind: "prompt",
  supplement: [Prompt],
) <intent_router_prompt_user>

#let clarify_query_prompts_system = read("code/clarify_query_system_prompt.py")
#figure(
  code()[
    #raw(clarify_query_prompts_system, lang: "python")
  ],
  caption: "Clarify query, system prompt",
  kind: "prompt",
  supplement: [Prompt],
) <clarify_query_system_prompt>

#let clarify_query_prompts_user = read("code/clarify_query_user_prompt.py")
#figure(
  code()[
    #raw(clarify_query_prompts_user, lang: "python")
  ],
  caption: "Clarify query, user prompt",
  kind: "prompt",
  supplement: [Prompt],
) <clarify_query_user_prompt>

#let geocontext_retriever_prompts_system = read("code/geocontext_retriever_system_prompt.py")
#figure(
  code()[
    #raw(geocontext_retriever_prompts_system, lang: "python")
  ],
  caption: "Geocontext retriever, system prompt",
  kind: "prompt",
  supplement: [Prompt],
) <geocontext_retriever_system_prompt>

#let infer_pages_prompts_system = read("code/infer_pages_system_prompt.py")
#figure(
  code()[
    #raw(infer_pages_prompts_system, lang: "python")
  ],
  caption: "Guidelines retriever, data extraction prompt",
  kind: "prompt",
  supplement: [Prompt],
) <guidelines_retriever_data_extraction>

#let guidelines_retriever_prompts_system = read("code/guidelines_retriever_system_prompt.py")
#figure(
  code()[
    #raw(guidelines_retriever_prompts_system, lang: "python")
  ],
  caption: "Guidelines retriever, system prompt",
  kind: "prompt",
  supplement: [Prompt],
) <guidelines_retriever_system_prompt>

#let infer_pages_structureless_prompts_system = read("code/infer_no_structure_system.py")
#figure(
  code()[
    #raw(infer_pages_structureless_prompts_system, lang: "python")
  ],
  caption: "Municipal ciztizen profile, data extraction prompt",
  kind: "prompt",
  supplement: [Prompt],
) <municipal_citizen_profile>

#let generate_answer_factual_system_prompt = read("code/generate_answer_factual_system_prompt.py")
#figure(
  code()[
    #raw(generate_answer_factual_system_prompt, lang: "python")
  ],
  caption: "Strategy planner, factual system prompt",
  kind: "prompt",
  supplement: [Prompt],
) <generate_answer_factual_system_prompt>

#let generate_answer_factual_user_prompt = read("code/generate_answer_factual_user_prompt.py")
#figure(
  code()[
    #raw(generate_answer_factual_user_prompt, lang: "python")
  ],
  caption: "Strategy planner, factual user prompt",
  kind: "prompt",
  supplement: [Prompt],
) <generate_answer_factual_user_prompt>

#let generate_answer_actionable_system_prompt = read("code/generate_answer_actionable_system_prompt.py")
#figure(
  code()[
    #raw(generate_answer_actionable_system_prompt, lang: "python")
  ],
  caption: "Strategy planner, actionable system prompt",
  kind: "prompt",
  supplement: [Prompt],
) <generate_answer_actionable_system_prompt>

#let generate_answer_actionable_user_prompt = read("code/generate_answer_actionable_user_prompt.py")
#figure(
  code()[
    #raw(generate_answer_actionable_user_prompt, lang: "python")
  ],
  caption: "Strategy planner, actionable user prompt",
  kind: "prompt",
  supplement: [Prompt],
) <generate_answer_actionable_user_prompt>

#let critic_answer_system_prompt = read("code/critic_answer_system_prompt.py")
#figure(
  code()[
    #raw(critic_answer_system_prompt, lang: "python")
  ],
  caption: "Strategy planner, system prompt",
  kind: "prompt",
  supplement: [Prompt],
) <critic_answer_system_prompt>

#let benchmark_system_prompt = read("code/benchmark_system_prompt.py")
#figure(
  code()[
    #raw(benchmark_system_prompt, lang: "python")
  ],
  caption: "Benchmark, system prompt",
  kind: "prompt",
  supplement: [Prompt],
) <benchmark_system_prompt>

// Table of acronyms, NOT COMPULSORY
#print-index(
  title: heavy-title(i18n(doc_language, "acronym-table-title"), mult: 1, top: 1em, bottom: 1em),
  sorted: "up",
  delimiter: " : ",
  row-gutter: 0.7em,
  outlined: false,
)

#pagebreak()

#heavy-title("Equations", top: 1em, bottom: 1em)

#highlight("TODO: CHECKER les maths encore une fois")
#highlight("TODO: juste citer la source wikipedia et enlever ça ???")
#highlight("TODO: EN FAIRE UNE FIGURE!!")
$
  overline(x) = frac(1, N)sum_(i=1)^N x_i && "where" overline(x) "is the sampled mean," N "the number of samples and" x_i "the" i_"th" "sample tile."
$ <sample_mean>

$
  s = sqrt(frac(sum_(i=1)^N(x_i-overline(x))², N-1)) && "where" s "is the sample standard deviation of a single tile."
$ <std>

$
  "Confidence interval for a single tile" & = overline(x) plus.minus t_(alpha/2, N-1) dot frac(s, sqrt(N)) \
                                          & "where" t_(alpha/2, N-1) "is the critical value from the"      \
                                          & "T-distribution for confidence level"
                                            1-alpha "and"                                                  \
                                          & N-1 "degrees of freedom."
$ <confidence_interval>

$
  "cosine similarity" & := cos(theta) = frac("A" dot "B", norm("A")norm("B")) = frac(sum_(i=1)^n A_i B_i, sqrt(sum_(i=1)^n A_i²) dot sqrt(sum_(i=1)^n B_i²))\
  &"where" theta "is the angle between A and B, two" n"-dimensional vectors"\
  &"and" A_i, B_i "the" i_"th" "components of vectors A and B."
$ <cosine_sim>

$
  "quartile coefficient of dispersion" & := frac(frac(1, 2)"IQR", frac(Q_3+Q_1, 2)) = frac(frac(1, 2)(Q_3-Q_1), frac(Q_3+Q_1, 2)) = frac(Q_3-Q_1, Q_3+Q_1)\
  & "where IQR is the interquartile range and" \
  & Q_1 "and" Q_3 "the first and third quartiles, respectively."
$ <qcd>

$
  "Spearman coefficient of correlation" & = frac("cov"["R"[X]", R"[Y]], sigma_("R"[X])sigma_("R"[Y])) \
  & "where R"[X] "and" "R"[Y] "are the ranks of raw scores" (X_i,Y_i), \
  &"cov"["R"[X]", R"[Y]] "is the covariance of the rank variables"\
  & "and" sigma_("R"[X]),sigma_("R"[Y]) "are the standard deviations of the rank variables."
$ <spearman>

#pagebreak()

// Table of listings
#table-of-figures()
