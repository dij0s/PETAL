#import "@preview/isc-hei-bthesis:0.5.3": *

#let doc_language = "en" // Valid values are en, fr

#show: project.with(
  title: "Optimisation de la planification énergétique urbaine par l'orchestration de l'IA",
  sub-title: "A Bachelor Thesis in Data Engineering", // Optional

  is-thesis: true,
  split-chapters: true,

  thesis-supervisor: "Jessen Page",
  thesis-co-supervisor: "Florian Desmons", // Optional, use none if not needed
  thesis-expert: "Nils Schüler", // Optional, use none if not needed
  thesis-id: "ISC-ID-25-8", // Your thesis ID (from the official project description) or none if not used
  project-repos: "https://github.com/dij0s/PETAL", // Your project repository


  school: "Haute École d'Ingénierie de Sion",
  programme: "Informatique et Systèmes de communication (ISC)",

  // Some keywords related to your thesis
  keywords: ("engineering", "data", "large language models", "AI agents", "energy planning"),
  major: "Data engineering", // "Software engineering", "Embedded systems", "Security", "something else"

  authors: "Dion Osmani",

  date: datetime.today(), // or datetime.today()
  language: doc_language, // en or fr
  version: "1.0", // or for instance "1.0", for the version of your thesis],
  code-theme: "bluloco-light",
)
// // If using acronyms
#import "@preview/acrostiche:0.5.2": *
#include "acronyms.typ"

// Let's get started folks!

#cleardoublepage()
#heavy-title("Abstract")
This work investigates the implementation and evaluation of an AI-powered solution to assist municipalities in energy planning.
As municipalities are required to develop energy planning documents to comply with federal and cantonal sustainability objectives and regulations, there is a growing need for tools that can process various energy sources of data and provide actionable insight to support human decision making.

The main objective is to assess the strengths and weaknesses of a multi-agent AI system that coordinates specialized AI agents through an orchestration layer to provide comprehensive energy planning assistance.
Public geospatial data sources and regulatory frameworks are leveraged to support recommendations.

It is evaluated using a dual approach: domain expert assessment and an automated LLM-as-a-judge benchmarking framework (G-eval) that measures performance across four criteria: data interpretation, methodology alignment, municipal relevance and technical compliance.

Results demonstrate that larger language models in key agents consistently outperform smaller ones across all evaluation criteria, with statistically significant improvements.
The system shows particular strengths in contextual reasoning and structured planning for actionable energy planning tasks.
However, the evaluation reveals significant limitations including unsupported claims, inconsistent data interpretation and high variability in automated scoring, indicating reliability concerns.

In conclusion, while the solution shows promise for assisting users in municipal energy planning by effectively retrieving, contextualizing and presenting geospatial data through natural language queries, it requires further refinement to achieve production-grade robustness.
The system demonstrates potential for supporting expert users in their planning processes but poses risks for uninformed users who may be misled by confident yet unsupported recommendations.

#abstract-footer("en")

#cleardoublepage()
#heavy-title("Résumé")

Ce travail étudie l'implémentation et l'évaluation d'une solution basée sur l'intelligence artificielle pour assister les communes dans la planification énergétique.
Les communes étant tenues de développer des documents de planification énergétique pour se conformer aux objectifs et réglementations de durabilité fédéraux et cantonaux, il existe un besoin croissant d'outils capables de traiter diverses sources de données énergétiques et de fournir des informations exploitables pour soutenir la prise de décision humaine.

L'objectif principal est d'évaluer les forces et faiblesses d'un système d'IA multi-agents qui coordonne des agents IA spécialisés à travers une couche d'orchestration pour fournir une assistance complète en planification énergétique.
Des sources de données géospatiales publiques et des réglementations sont exploités pour soutenir les recommandations.

Elle est évaluée selon une approche duale: évaluation par des experts du domaine et un framework d'évaluation automatisée utilisant un LLM comme juge (G-eval) qui mesure la performance selon quatre critères: interprétation des données, alignement méthodologique, pertinence communale et conformité technique.

Les résultats démontrent que les modèles de langage plus grands dans les agents majeurs surpassent constamment les plus petits sur tous les critères d'évaluation, avec des améliorations statistiquement significatives.
Le système montre des forces particulières dans le raisonnement contextuel et la planification structurée pour les tâches de planification énergétique actionnables.
Cependant, l'évaluation révèle des limitations significatives incluant des affirmations non supportées, une interprétation irrégulière des données et une haute variabilité dans la notation automatisée, indiquant des inquiétudes quant à la fiabilité.

En conclusion, bien que la solution montre du potentiel pour assister les utilisateurs dans la planification énergétique communale en récupérant, contextualisant et présentant efficacement des données géospatiales à travers des requêtes en langage naturel, elle nécessite un raffinement supplémentaire pour atteindre une robustesse apte à la production.
Le système démontre un potentiel pour soutenir les utilisateurs experts dans leurs processus de planification mais pose des risques pour les utilisateurs non informés qui pourraient être induits en erreur par des recommandations confiantes mais non supportées.

#abstract-footer("fr")

#cleardoublepage()
#heavy-title("Artificial Intelligence notice")

While generative artificial intelligence is the core of this work, it has also been a great help in assisting the following tasks: (1) prompt engineering and (2) thesis report rewording#footnote("The language models used for these tasks are GPT-4.1 (OpenAI) and Claude 3.7 (Anthropic).").

The different prompts defined in the appendix were refined with the use of generative artificial intelligence. This strongly enhances the quality of the prompts, implementing prompt engineering techniques to ensure clear, concise and effective instructions.

#ref(<prompt_creation>) defines typical directives that were given to the language model.

On top of that, sentences in the following report were often revised with the assistance of a language model, improving their clarity and readability.

Besides that, all other aspects of this work are my own.

#cleardoublepage()
// Get the proper title for acknowledgements if not written in English
#heavy-title(context i18n(inc.global-language.get(), "acknowledgements"))

Thank you to Jessen Page, Florian Desmons and Cédric Travelletti for their expertise and insight into this work.
#v(2cm)
And to those who sacrificed so much for us, THANK YOU.

#table-of-contents(depth: 2)
// Enable headers and footers from this point on
#set-header-footer(true)

= Introduction
Human-driven activities are the principal cause of climate change and global warming #ref(<EvidenceNASAScience2022>).

Domain experts have monitored this matter and proposed various frameworks to address and mitigate these problems. In Switzerland, these different frameworks are implemented in the legislation and guidelines (at federal and canton levels) to steer the country towards a more sustainable future.
Municipalities may introduce in their regulations sustainability requirements that are more constraining than those set by the cantonal law #ref(<RS7301Loia>, supplement: [article 12, al. 5]).

== Context

Municipalities in Switzerland are required to submit an energy planning document which outlines their future strategies to comply with those directives while also considering the characteristics of their energetical landscape.

These different properties can be quantified and analyzed through the use of very valuable resources: data.
Data are emitted by various sources; sensors, energy models or citizens' records for example all yield data points that help us assess different indicators we are willing to measure against our municipality. These indicators, _in fine_, help us evaluate our progress towards energy-related goals.

Over the past few years, #acr("AI") has rapidly transformed our habits when interacting with information.

== Problem

#acrpl("LLM") allow users to interact with these systems using natural language, facilitating the interface between humans and _machines_. They can provide insights into vast amounts of data at speeds and scales beyond human capabilities.

As a result, this work tackles the implementation of a solution which assists users into energy planning for a municipality.

This complex problem is approached by leveraging the power of _specialized_ AIs which each offer expertise into a variety of domains and are coordinated using an _orchestration_ AI to provide a solution. The expertise of the user interacting with the system tailors this solution to the specificities of the municipality.

Key engineering steps in this implementation are: (1) identifying the primary information and data sources that are relevant to this process, (2) structuring the different AIs into an architecture whose components and interfaces are well-defined and (3) ultimately implementing the solution.

== Objectives

The main objective of this thesis is to investigate how effective and reliable such a solution is and to assess its strengths and weaknesses. Additional goals are also outlined:
- Defining the important information in assisting user decision making.
- Understanding which data sources are available and relevant for this decision making.
- Structuring the decision by using an orchestration AI and many specialized AIs.
- Training specialized AIs on specialized datasets.
- Simplifying the user interface by handling the communication between the orchestration AI and the specialized AIs.

The project is scoped to municipalities within the canton of Valais/Wallis and strictly relies on publicly available data.

A user-friendly interface enables users to interact with the system in a conversational manner and visualize a map of the municipality with different layers, as depicted in #ref(<petal_interface_pv>) and #ref(<petal_interface_water>).

#figure(
  image("figs/petal_interface_pv.png", width: 100%),
  caption: "Web interface displaying a report on photovoltaic solar energy potential in Sion. The interface features a chat panel for querying the system and an interactive map, visualizing rooftop-level solar energy data.",
)<petal_interface_pv>

#figure(
  image("figs/petal_interface_water.png", width: 100%),
  caption: "Web interface displaying a follow-up report on small hydropower potential in Sion. The interface includes a chat panel showing system responses and an interactive map, visualizing water flow data and rooftop-level solar energy.",
)<petal_interface_water>

Finally, user behavior is analyzed to takeaway user preferences which gradually adapt the answers to better meet the user expectations.

This document outlines the methodology used to design and implement the solution, details the evaluation framework and presents the results, in relation to the research question.

= State of the Art <state_of_art>

Before exploring the methodological approach of the solution, it is necessary to introduce and understand the key concepts of the underlying technologies.

On top of that, related work in the field of AI-assisted urban planning is presented, positioning this thesis within the broader landscape of such applications.

== Technological Background

This section offers a broad overview of the technological background, focusing on the clarification of the fundamental concepts.

#acrpl("LLM") represent a major breakthrough in the field of natural language processing #ref(<vaswaniAttentionAllYou2023>).
These models, at their heart, are designed to predict the next word in a sequence, based on previous ones.

By processing sequential text data, they are able to understand context and generate coherent text in response.
Modern language models are trained on large amounts of text data prior to the integration of human feedback, aligning their outputs with human preferences #ref(<ouyangTrainingLanguageModels2022>).

Since their introduction to the public, significant evolutions in their architectures and capabilities have been observed.

Notably, #acrpl("RLM") enable more systematic problem-solving and logical deduction by incorporating explicit reasoning steps into their generation process, at the cost of increased computational resources and latency.

Language models are available in any shape, form and size, ranging from small to large models. Their number of parameters range from a few million to hundreds of billions for the largest.
More parameters enable more complex and nuanced language understanding and generation, at the cost of greater computational demand.

#acr("RAG"), on the other hand, addresses a fundamental limitation of language models #ref(<lewisRetrievalAugmentedGenerationKnowledgeIntensive2021>). As LLMs are trained on static datasets, they lack knowledge in domain-specific information.

Therefore, RAG systems overcome this limitation by combining the generative capabilities of these models with external knowledge retrieval, enabling them to ground their responses in up-to-date, factual information.

The RAG pipeline consists of three components:
1. Retrieval: relevant documents or data points are identified using semantic similarity search, implemented through vector embeddings.
2. Augmentation: retrieved information is incorporated into the context of the model.
3. Generation: responses are produced based on both the original query and retrieved knowledge.

Vector embeddings play a crucial role in these implementations, transforming documents into high-dimensional mathematical representations that capture semantic meaning and enable efficient search #ref(<mikolovEfficientEstimationWord2013>).

Furthermore, AI agents represent autonomous software systems that perceive their environment, make decisions and take actions to achieve specified goals.
In the context of LLM-based applications, these agents leverage natural language understanding to perform complex multi-step tasks with minimal to no human intervention.

Multi-agent systems coordinate multiple specialized agents, each responsible for specific tasks of a broader problem.
In response, orchestration frameworks have emerged to manage the complexity of these solutions, providing abstraction to define agent workflows.

With the technological foundation established, the next section reviews related work to situate this thesis within the context of current research and applications.

== Related Work

The AI Institute at ITMO University published a paper in 2025 titled _LLM Agents for Smart City Management: Enhancing Decision Support Through Multi-Agent AI Systems_ #ref(<kalyuzhnayaLLMAgentsSmart2025>). The study examines how the natural language processing strengths of LLMs, combined with the distributed problem-solving abilities of multi-agent systems, can enhance urban decision-making processes.

The research focused on the testing of three hypotheses: (1) evaluating the capability of LLM agents to effectively route and process diverse urban queries against existing urban information systems, (2) the effectiveness of #acr("RAG") technology in improving response accuracy when working with local knowledge and regulations and (3) the impact of integrating LLM agents with existing urban information-systems - increasing efficiency and decreasing the decision making process time.

Their proposed solution was tested against 150 question-answer pairs and used St. Petersburg's Digital Urban Platform as a testbed.
The testing dataset was curated and built by a group of human experts such as specialists in urban data analysis, GIS specialists and urban architects.

They then evaluated different configurations of LLM agents and state-of-the-art models against two primary metrics: G-eval and answer relevance.
The G-eval metric provides greater compliance with human requirements as it uses LLMs to evaluate answers from other LLMs based on custom user criteria.

These criteria can for e.g. be provided as a list of rules specifying precise steps the LLM should take for evaluation, mirroring human reasoning process.
The answer relevance metric, on the other hand, assesses the relevance of the answer from the LLM when compared with the correct answer provided by the experts. This process also leverages LLMs as it first extracts the different statements from the answer and then compares those to the reference answer.
Both metrics are bounded between 0 and 1, where a higher value indicates better performance.

In summary, the results show greater performance when integrating the RAG technology and urban information-systems to the solution (G-eval scores of 0.68-0.74) compared to standalone LLM responses (G-eval scores of 0.30-0.38).
Relevance scores, on the other hand, remain high whatever the configuration as they are inherently designed to produce semantically relevant text.
They also concluded that this research proved practical real-world city management application as it enables efficient processing of urban planning tasks while maintaining high relevance in responses and shortening task completion time from days to hours.

The ITMO study presents a research-driven implementation of LLM agents focusing on decision support through integration with urban data platforms which curate and process urban data to provide insights and recommendations for urban planning and management.

Rather than relying on these large-scale platforms, the present work explores the potential of leveraging publicly available data from federal and cantonal sources while also considering the interface with municipal archives and municipal resident records.

This work strongly values the user experience with efforts in enhancing conversational interactions through preference-driven reporting and improving the clarity and quality of reported decisions.
It neither serves as a continuation nor a re-implementation of the ITMO study, but rather represents an independent application of AI agents to a related use case specifically adapted to the context of a Bachelor's thesis and shaped by my practical implementation choices and problem-solving approach.
Any solution designed and implemented around similar goals and data-related constraints, regardless of the specificities of the use case, may result in an architecture that is somewhat similar.

As we forget about the use case and consider more generic research on the matter, we encounter  publications that rather focus on the optimization of various aspects of AI agents, such as their scalability, efficiency and robustness. While this thesis tackles some of these challenges, it does not incorporate major research efforts into these areas.

Another AI-centric solution relevant to this use case is aino#footnote("https://www.aino.world/"), described on their homepage as an _AI GIS Analyst for Urban planning teams_. Developed and marketed in the United States, it is a commercial business solution offering a platform where an AI analyzes sites and provides visual insights from simple questions. This solution inspired me into a few design and user experience improvements in my work as no implementation details are provided.

These two points of view position this project within the fast-changing field of AI-driven solutions for urban planning and beyond.

= Methodology <methodology>

The following chapter provides an overview of the methodology that has been adopted in this thesis and describes the approach taken to design, implement and further evaluate the proposed solution.
A clear emphasis is put onto the key decisions that have shaped the solution throughout its development as this chapter aims to offer full transparency and reproducibility which enables readers to understand the rationale and structure behind it.

== Requirements

Identifying and describing the primary requirements lays the groundwork for the design of the solution.
The main requirements were established and further refined as the project progressed to meet the expectations and project goals throughout ongoing discussions with the supervisors.

Those are categorized into functional and non-functional requirements. Functional requirements focus on user needs and product features whereas non-functional requirements focus on user expectations and product properties.

The first step is to understand the problem that is solved. Energy planning typically involves:
- Identifying available energy resources#footnote("The resources and needs are assessed within the geographical boundaries of the municipality, only."), infrastructure and untapped potential.
- Characterizing the needs.
- Assessing different measures and their impacts; sobriety (reducing energy consumption), efficiency (more efficient technologies) and production of renewable energy sources.

Hence, assisting users in energy planning requires a solution that can gather relevant data sources, analyze and present the current energy landscape of the municipality and provide actionable recommendations tailored to the context of the municipality while ensuring compliance with the legislation and guidelines it is subject to.

Besides that, it had been requested that the user interface has a map showcasing the assessed data points within the municipality as well as for the AI to be able to _remember_ the user's preferences and past interactions to have the answer better fit what the user expects.

These requirements are summarized in the #ref(<requirements_table>).

Ongoing weekly evaluations with supervisors led to the identification of additional needs, which progressively refined and improved the system to better support energy planning and enhance user experience.

As these requirements established the basis for the project, the next section presents the design and implementation details of the solution.
#pagebreak()
#show table.cell.where(y: 0): strong
#show figure: set block(breakable: true)

#set table(stroke: (x, y) => {
  if x == 0 {
    (right: 0.7pt + black)
  }
  if y == 0 {
    (bottom: 0.7pt + black)
  }
  if y == 9 {
    (bottom: 0.3pt + black)
  }
})
#set table(
  fill: (x, y) => { if calc.odd(y) { rgb("D8E1E6") } },
  align: (x, _) => if x == 0 { center } else { left },
)
#figure(
  rotate(
    -90deg,
    reflow: true,
    table(
      columns: (12%, 25%, 58%),
      table.header([Type], [Requirement], [Description]),
      [Functional],
      [Multi-agent orchestration],
      [The system must coordinate multiple specialized AIs through an orchestration layer to break down complex tasks into manageable subtasks.],

      [Functional],
      [Conversational interface],
      [The system must provide a natural language interface allowing users to interact with the AI through conversational queries.],

      [Functional],
      [Gather relevant data sources],
      [The solution must collect and integrate data from various sources relevant to the municipality's energy landscape.],

      [Functional],
      [Provide actionable recommendations],
      [The solution must generate tailored recommendations for energy planning, complying with the guidelines that apply to the municipality.],

      [Functional],
      [Analyze and present energy landscape],
      [The system must process, analyze and present the current energy landscape of the municipality, enabling users to understand available resources and needs.],

      [Functional],
      [Municipality map],
      [The user interface must include a map displaying assessed data points within the municipality for better spatial understanding.],

      [Functional],
      [Preference memory],
      [The AI must remember user preferences and past interactions to adapt responses and improve user experience over time.],

      [Functional],
      [Live feedback],
      [The system should perform a binary quality check on responses using a set of quality criteria and automatically regenerate the response if the conditions are not met.],

      [Functional],
      [Data transparency],
      [The solution should be transparent and provide the source and data processed from datasets, increasing user trust in the solution.],

      [Non-functional],
      [Extensibility and adaptability],
      [The architecture must allow for the addition of new features and data sources as requirements evolve.],

      [Non-functional],
      [Usability and accessibility],
      [The interface must be intuitive and accessible for users with varying technical expertise.],
    ),
  ),
  caption: "Table outlining the system requirements for the solution, divided into functional requirements (e.g., data integration, conversational interface) and non-functional requirements (e.g., usability, extensibility)",
) <requirements_table>

#pagebreak()
== System Design <system_design>

A modular, scalable and adaptable architecture was designed to ensure that the solution can adapt to evolving requirements and facilitate future enhancements. The system is organized into distinct components which are all responsible for a specific set of functionalities and ensure clear separation of concerns and ease of maintenance.

The following chapter covers both the design and the implementation aspects of the solution.

#figure(
  image("figs/system_design_global.svg", height: 5cm),
  caption: "High-level architecture diagram of the system showing the three-layer design and components with (1) the frontend layer (user interface), (2) the backend layer (AI agent orchestration and business logic) and (3) the external services layer (third-party APIs and data sources).",
)<global_system_design>

The global architecture in its most simplified form is presented in the #ref(<global_system_design>). The system is broken down into three distinct layers:
- The frontend layer manages user interaction and presentation of data, providing an intuitive interface for users to communicate with the system.
- The backend layer is responsible for business logic, orchestrating AI agents, processing data, managing a database and handling requests from the frontend.
- The external services layer provides access to third-party #acrpl("API"), a set of protocols and tools that allows different software components to communicate with each other, enabling the system to retrieve data from external platforms and services.

The layers are made up of various components. The basic dataflow between them is presented in the #ref(<global_dataflow>).
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
  caption: "Table describing the data flow within the high-level components. It outlines how user queries are handled by the AI agent, gathering information from a local database and external APIs before returning a processed response to the user interface.",
) <global_dataflow>

The nature and type of data that are exchanged between the AI agent and both the local database and third-party APIs are discussed in each individual segment, dedicated to the respective components.

Components are designed as modular and independent services, each containerized using Docker#footnote("https://www.docker.com/"), a solution first released in 2013 by Docker Incorporated.

This approach introduces a lightweight containerization paradigm where each container is isolated from the host system and other containers, providing a consistent runtime environment.

As such, the solution is easily portable from development to production.

Now that the main components have been introduced, the following section describes the design and implementation of the AI system.

=== AI agent

Recently, the term "AI agent" has become a buzzword, leading to frequent misuse.
Most definitions (#ref(<WhatAreAI2024>), #ref(<WhatAreAI>)) agree that the key behavior that distinguishes AI agents from other solutions is their degree of autonomy as they are able to operate and make decisions independently to achieve a set goal.

The complete solution, whose sole task is urban energy planning, aligns with this definition, as do each of its individual components. Therefore, the term "AI agent" will be used to describe both the entire system and its subsystems.

Human conversations rely on context and prior knowledge and so does the system's architecture. To deliver a conversational experience, it is essential that the architecture is built to effectively preserve and re-use this context throughout the discussion.
One might suggest that the straightforward approach to maintaining conversational context is to include all previous exchanges with the user. However, this method can be very inefficient and comes at great cost.

LLMs rely on a self-attention mechanism to identify and concentrate on the most relevant parts of the input sequence. Each token, the basic unit of text (word, single character, group of words, etc.) that is processed by the model, is assigned a weight reflecting its importance. This allows the model to prioritize relevant information and ignore irrelevant details.

Hence, when the entire conversation history is provided to the model, important tokens may get lost in the larger context and potentially lead to incorrect responses (phenomenon called attention diffusion).

Considering this, a more efficient approach is proposed, relying on a single key assumption: each conversation focuses exclusively on energy planning for one municipality at a time.
Accordingly, the conversational context is modeled as a single object that is updated at every turn. It is defined in the #ref(<conversational_state>).
#let destructured_state = read("code/destructured_state.py")
#figure(
  code()[
    #raw(destructured_state, lang: "python")
  ],
  caption: "Python data structure representing the conversation state, including user inputs, language, intent, location, query aggregation and flags for clarification or memoization. It also stores contextual tools and constraints for guiding energy planning conversations.",
) <conversational_state>

The different agents leverage #acrpl("RLM"), breaking problems into logical steps and mimicking human reasoning.
Compared to standard language models, they are particularly valuable for tasks that require logical deduction and planning but come with notable drawbacks as they are typically more computationally intensive, leading to higher operational costs and increasing latency in response times.

To address this challenge, the architecture is intentionally designed to reduce the number of language model calls whenever possible.
Although these models show very strong capabilities, their usage can be excessive for certain tasks and lead to increased costs and latency.
#pagebreak()
By constraining the conversational state and narrowing the scope of each agent, it is also possible to reduce the computational load and latency by simply swapping out these large reasoning models by smaller, better-suited models.

Doing so, it becomes possible to select reasoning models that are better suited to specific tasks while minimizing the computational costs.

#figure(
  image("figs/ai_agent_system_design.svg", width: 100%),
  caption: "Architecture of the AI multi-agent system showing the workflow between specialized agents. The diagram illustrates the Intent Router, Clarify Query, Geocontext Retriever, Guidelines Retriever, Strategy Planner and Critic agents with their interconnections. Solid arrows represent mandatory transitions while dashed arrows indicate conditional transitions based on the evolving conversational state.",
)<ai_agent_design>

The architecture in the #ref(<ai_agent_design>) is modeled after a #acr("FSM"), where each node represents an agent and each edge represents a transition that is either always executed (solid) or conditionally executed (dashed). The dynamic flow of control between agents is guided by the evolving conversational state. It is finite, per definition, as the state takes value in a discrete set.

On the implementation-side, LangGraph#footnote("https://www.langchain.com/langgraph"), an open-source Python framework, is used to implement the AI agent architecture. Unlike linear pipelines, LangGraph uses a graph abstraction by default, which is particularly well-suited for this state machine architecture.

This graph-based structure brings determinism to the system’s behavior as the flow between agents is defined by the architecture itself, rather than being dynamically determined by agent-to-agent conversations as in frameworks like Microsoft's AutoGen#footnote("https://www.microsoft.com/en-us/research/project/autogen/").

Another framework that had been considered was PydanticAI#footnote("https://ai.pydantic.dev/") which offers a structured, type-safe approach to building agent systems by leveraging Pydantic models for inter-agent communication and behavior definitions. However, it lacks the built-in support for complex state transitions.

All of the available multi-agent AI frameworks are relatively novel and in constant evolution. LangGraph benefits from being built on top of the already renowned LangChain#footnote("https://www.langchain.com/") ecosystem which adds to its reliability and ease of integration with other technologies.
Pydantic’s type safety will still be implemented within the project to enhance data validation and error handling.

Naturally, the framework requires integration with a language model provider to operate effectively and exploit their capabilities.

The system leverages Ollama#footnote("https://ollama.com/"), a lightweight, open-source platform designed for running language models on a local machine.
It offers a wide range of pre-built models, facilitating the integration and experimentation with models of different sizes and capabilities.

While other solutions such as VLLM#footnote("https://docs.vllm.ai/en/latest/") may offer greater performance and reduce overall latency in the application, Ollama stands out with its support for hot-swapping models, a feature that enables the dynamic switching between models, _on-the-fly_.
This is particularly valuable in the scenario of multi-agents architectures, where different agents may need models with specific abilities.

On the counterpart, only open-source models are supported, naturally limiting the selection of models, although it also prevents vendor lock-in.

Overall, it provides an easy-to-use and flexible approach for experimenting with language models.

Finally, the choice of database, in the system architectures (#ref(<global_system_design>) and #ref(<ai_agent_design>)) is primarily influenced by the nature of the data to be stored and the more general use-case.

In this work, the database is used to store and retrieve structured documents. It holds user preferences, usage statistics and vector embeddings for documents, enabling the retrieval of semantically related information.

The choice was made towards Redis OSS#footnote("https://redis.io/"), an open-source, in-memory data store especially known for its versatility and performance.
Redis supports a variety of data structures, including documents which provide a flexible schema for structured data and the newly introduced vectors.

Its memory-first approach and indexing capabilities enable low latency, ideal for multi-agent systems where information must be retrieved in real-time.

Unlike more specialized databases like Pinecone#footnote("https://www.pinecone.io/") or Qdrant#footnote("https://qdrant.tech/qdrant-vector-database/"), Redis provides a unified, mature and versatile architecture suitable for multiple purposes.

Now that the general technological tools have been covered, the next sections provide an in-depth explanation of the multi-agent architecture.

Before that, the main responsibilities of each agent are as follows:
- The _intent router_ routes the user's query to the appropriate agents and accumulates query context.
- The _clarify query_ clarifies the user's query if it is ambiguous or incomplete.
- The _geocontext retriever_ retrieves the geospatial data relevant to the request.
- The _guidelines retriever_ retrieves the relevant energy planning guidelines relevant to the query.
- The _strategy planner_ plans the energy strategy based on the retrieved data and guidelines.
- The _critic_ evaluates the proposed energy planning strategy and possibly restarts the whole process.

With the overall solution defined, the following sections dig in the details of each agent and their implementation.

==== Intent Router <intent_router>

The intent router is a crucial component of the solution. It is the entrypoint of the system and orchestrates the different agents.

Upon receiving a user prompt, the agent analyzes the query to extract its underlying intent. This involves identifying these elements:
- The intent: specifies whether the query is "factual" (for e.g. requesting data) or "actionable" (seeking planning guidance, recommendations, or strategic advice).
- The location: the municipality name mentioned in the user request, if available.
- The aggregated query: a summary that combines all available context from the current conversation and the previous query into a single one.
- The conversation type: identifies the conversational context; "new_analysis" (fresh query), "correction_request" (user questions the accuracy of a previous response) or "follow_up" (user requests additional detail or expansion on the same topic).
- The need for clarification: defines whether more information is needed to understand what users wants (e.g., missing location, unclear intent, or vague request).
- The needs for memoization: specifies if the user provided explicit preferences, corrections to assumptions, or scope refinements that should be remembered for future queries (for e.g. the format used to summarize the retrieved data or only considering a single aspect from certain data points).

Implementation-wise, the output of the language model is constrained to a single Pydantic#footnote("https://docs.pydantic.dev/latest/") data schema.
While language models typically generate natural language responses, these complex multi-agent systems benefit from a structured output format that can easily be further processed.

This is possible thanks to OpenAI#footnote("https://openai.com/"), introducing the support for structured outputs in late 2024, a feature that has since been adopted by many providers. Pydantic enables this by providing a simple way to define data models and validate data, ensuring the output format is consistent.

The instructions in #ref(<intent_router_prompt_system>) and #ref(<intent_router_prompt_user>) are designed to guide the language model on how to generate the correct output#footnote("Please note that all prompts are included in the appendix.").
Few-shot prompting helps the language model by providing examples on how to complete the task, helping it generalize to subsequent prompts.

In this context, it facilitates the definition of the _aggregated_query_ and _needs_memoization_ fields by providing a set of prompts with the result that would be expected, in return, for both of them.

All these fields except the aggregated query take value in a finite set of options (considering the _location_ field is either set or unset.).
Consequently, it is very easy to plan and orchestrate the following actions.

Since the application must offer a conversational experience, the previously determined state is updated and not overwritten.
Past fields are only updated if they differ from the new ones. As such, context and knowledge is properly accumulated over time.

Once the municipality is provided, for example, it is not needed anymore as the request is assumed to concern the same municipality (#ref(<example_conversation>)).

On the other hand, when a request treats a different municipality, both the _context_tools_ and _context_constraints_ defined in #ref(<conversational_state>) are reset. This way, the data associated with the previously discussed municipality are cleared.
To ensure correct implementation, whenever a request concerns a different municipality, both the _context_tools_ and _context_constraints_ defined in #ref(<conversational_state>) are reset. This means the geospatial data and guidelines previously associated with the conversation are cleared.

On top of that, user-provided feedback and corrections shape the system's behavior allowing it to adapt to the user's preferences.
When there is a need for memoization, the system stores both the previous query (the _corrected_) and the current query (the _correctee_), in the database.

This highlights the interfaces of the intent router, as detailed in the #ref(<intent_router_design>).
#figure(
  image("figs/intent_router_design.svg", width: 12cm),
  caption: "Interface diagram for the Intent Router agent showing input/output data structures and interaction with the local database. The input is the prompt, while output provides structured classification that guides subsequent agent selection in the workflow.",
)<intent_router_design>

An assumption still lies in the nature of the field _location_ as it is assumed to either be set or unset. A set location does not necessarily imply that it is a valid municipality, inscribed in the published Swiss official commune register#footnote("https://www.bfs.admin.ch/").
A solution is proposed in the #ref(<geocontext_retriever>).

Finally, the query is routed as shown in the #ref(<ai_agent_design>).
- If clarification is needed (either because the need for clarification is explicitly requested, or fields are missing), the request is sent to the clarify query agent (2).
- If the conversation type is a correction request, the query is sent to the geocontext retriever agent (4) as it implies a re-assessment of the response without extra information.
- If the intent is said to be actionable, the request is sent to both the geocontext retriever and the guidelines retriever agents, concurrently -(4) and (5)- enabling guideline-compliant responses.
- Otherwise, the query is sent to the geocontext retriever agent (4).

With the aim of the user's query now clearly defined, the next step is to address any ambiguities or missing information with the clarification agent.

==== Clarify Query
Clarifying and resolving vagueness in the user's query is essential to better understand the fundamental intent and provide an aligned response.

With the output of the intent router agent properly defined, the two cases which lead to the need for clarification are either an explicit request for clarification due to ambiguity or missing information.

Those two cases are both handled at once as a language model is prompted with the user's query and missing fields to generate and stream a response inquiring for further information or clarification (#ref(<ai_agent_design>), transition 3). The interfaces are illustrated in the #ref(<clarify_query_design>).
#figure(
  image("figs/clarify_query_design.svg", width: 12cm),
  caption: "Interface diagram for the Clarify Query agent showing input/output data structures. The input is the conversation state and the output is the generated clarification question to guide further user input.",
)<clarify_query_design>
#pagebreak()
In the following turn, the newly provided information is merged with the previously deduced intent as designed and presented in the #ref(<intent_router>).

This task is supported by both #ref(<clarify_query_system_prompt>) and #ref(<clarify_query_user_prompt>).

With no _structural_ ambiguity left in the user's query, the intent router agent can now proceed to route the query to the geocontext retriever and guidelines retriever agents.

==== Geocontext Retriever <geocontext_retriever>

Energy planning as defined in the solution requires assessing the energy resources, infrastructure, potential and needs within the municipality. The geocontext retriever is responsible for this task.

Before profiling the municipality, it is essential to identify the different public data sources that are available.
Throughout its federal and cantonal institutions, Switzerland provides a wide range of public data such as GeoAdmin#footnote("https://www.geo.admin.ch/en"), the geographic information platform of the Confederation which offers direct access to geospatial data and maps.

The data originate from various offices commissioned by the Confederation:
- Swiss Federal Office of Energy (SFOE)
- Federal Office for Spatial Development (ARE)
- Federal Office of Topography (swisstopo)
- Federal Office for Agriculture (FOAG)
- Federal Office for the Environment (FOEN)

The GeoAdmin API#footnote("https://api3.geo.admin.ch/index.html") provides a standardized interface for querying and manipulating geospatial data and relies on fair usage policies (20 requests per minute on a 24/7 average).
The datasets are also available for download.

The choice has been made to use the GeoAdmin API instead of downloading and maintaining local datasets as it ensures (1) that the data are always up to date and (2) removes the need for additional setup and maintenance of a dedicated geospatial database, a task that is particularly time-consuming in such a short time frame.

In a real-world scenario, exploiting data locally allows for preprocessing and aggregation which significantly reduces latency during user interactions.

Mechanisms such as caching and geospatial indexing, a technique used in databases enabling quick identification of objects located within a particular geographical area, would be useful for greater scalability of the solution.
On the other hand, periodic updates would be necessary to ensure the data remains up to date.

Although the solution does not require such implementations, it remains essential to understand how geographical data is structured and the key concepts behind it.

Datasets are labeled as layers, as the data are organized according to the geospatial paradigm. Data are discretized into points, meshes, polygons and other spatial representations, all collectively referred to as features.

Those features are independent geometries located in the space, without inherent relationships.
Thus, identifying relevant features within a municipality implies searching them inside its geographic boundaries, since no relation lies between these entities.

Although the GeoAdmin API enables searching for features in a given area, it is subject to a maximum number of 50 features retrieved per request.
Consequently, identifying them requires breaking down the search area into smaller sub-areas and querying each sub-area separately.

This has been implemented by first clipping settlements and centres of larger cities onto the municipality's geometry, optimizing the search area and applying a spatial tiling on top. Different layers obviously require different tiling sizes, depending on the number and resolution of features.

The #ref(<datasets_table>) presents the datasets incorporated in the solution and retrieved from the GeoAdmin API.

Leveraging these datasets provides the necessary information to assess the energy needs, potential and infrastructure within the municipality and establish a baseline profile for energy planning.
The discretization of the different data sources showcases the importance of spatial tiling when dealing with these data.

In the average municipality, certain features are few and easily assessable whereas it is impossible to retrieve meaningful insights from the greater-resolution datasets (for e.g. the suitability of roofs, per roof pane) without additional processing or aggregation.

Therefore, the features within the municipality are (1) identified within the municipality, (2) aggregated to the municipality level and (3) brought back to the same GWh/year unit#footnote("Only applies to energy measures. Energy is deduced from power, in watts, assuming non-stop operation (24/7/365)."). This standardization allows for an easier and consistent comparison of scalars, on a yearly basis, crucial for interpretability but comes with drawbacks:
- The information of variability within the municipality itself is lost.
- The aggregation of data is a costly process, especially when doing this on the fly.

The first issue is partially recovered from the fact that the layers are displayed at their original resolution, in the web interface. This way, the variability can be easily visualized.

The second issue is mitigated depending on the nature of the data. The basic approach to aggregation requires the summation of values and is only needed for datasets that benefit from great precision.

On the other hand, datasets that do not require such precision are subject to statistical estimation:
1. The spatial tiling is randomly sampled.
2. The features within the sampled tiling are identified and their values summed.
3. The sample mean (#ref(<sample_mean>)) and standard deviation (#ref(<std>)) are calculated.
4. The confidence interval is computed using a T-distribution and confidence level (#ref(<confidence_interval>)).

This random geographical sampling process is depicted in the #ref(<sampling_design>).
#figure(
  image("figs/tiling_design.svg", width: 100%),
  caption: "Illustration of the random geographical sampling methodology applied to the municipality of Grône. The diagram shows the statistical approach used to estimate spatial data from high-resolution geospatial datasets by randomly sampling spatial tiles within municipal boundaries.",
)<sampling_design>

Choosing the sampling size and confidence level is important for a proper statistical estimation. In this project, both parameters are set empirically and kept relatively large to benefit from lower computational costs, but without optimizing for the best possible accuracy.

As such, only the suitability of roofs and facades for use of solar energy is estimated using this technique as they are both datasets which showcase potential of exploitation rather than precise measurements on top of being well distributed in the geographic space. The confidence level in this case is set to 80%.

With the data standardized and properly aggregated, the geocontext retriever agent must now be able to interact with them.

Previously, AI agents were described as autonomous systems able to operate and make decisions independently. These operations rely on tools.

A construction worker, for example, has different tools for different needs, such as a hammer for nails, a saw for cutting wood and a level for ensuring straight walls. The tools come with a set of instructions describing how to use them and what to expect from them.

The same applies to AI agents, which have, in this work, different tools allowing them to query, aggregate and retrieve data from the datasets in #ref(<datasets_table>).
Consequently, language models can leverage their natural language processing capabilities to choose the appropriate tools for the query.

An issue with the current approach is that when the _toolbox_ is too large, it becomes difficult for the language model to choose the right tools for the job.

This issue is addressed by exploiting the power of embeddings, enabling the system to easily retrieve the different tools from their embedded descriptions, semantically. On top of that, it is more efficient computation-wise than prompting the language model to choose them.

When retrieving tools, the system computes the cosine similarity between both embeddings to quantify the semantic similarity (#ref(<cosine_sim>)).
Finally, the quartile coefficient of dispersion is measured against the distribution of retrieved similarity scores (#ref(<qcd>)).

This indicator provides a measure of the uniformity of the retrieved tools that is less sensitive to outliers than measures such as the coefficient of variation.
As such, if the distribution of scores is too uniform, the tools are provided to a language model that leverages its capabilities to only distinguish the appropriate ones, in the given context (#ref(<geocontext_retriever_system_prompt>)).

This approach reduces the overall computational cost while increasing the quality of tool selection.

With the appropriate tools chosen, the system can effectively retrieve the data. It is simply added to the _context_tools_ field in the conversational state (#ref(<conversational_state>)), as presented in the #ref(<geocontext_retriever_design>).
#figure(
  image("figs/geocontext_retriever_design.svg", width: 11cm),
  caption: "Interface diagram for the Geocontext Retriever agent showing input and output data structures. The input is the conversational state and the output is the updated state with the retrieved geospatial data. It also features the interaction with geospatial APIs to obtain relevant geographical information.",
)<geocontext_retriever_design>

Expanding the functionnalities of the geocontext retriever is straightforward: new data sources, APIs or even simulation models can be integrated as additional tools the agent may use.

Geospatial information is accumulated over the conversation turns, allowing for context-aware planning and consistent, spatially informed decisions. It is only reset when switching to a new municipality as it becomes invalid.

In the #ref(<intent_router>), the validity of the location is not confirmed. This is directly implemented in the different tools and routing of this agent (#ref(<ai_agent_design>)).
- If the location is incorrect, retrieving data raises an error and the request is routed to the clarify query agent (6).
- Otherwise, the query is sent to the strategy planner agent (7).

Once the relevant data are gathered, the next stage is for the strategy planner agent to analyze this information to conduct proper planning.

#set table(
  fill: (x, y) => { if calc.odd(y) and x != 0 { rgb("D8E1E6") } },
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
#figure(
  rotate(
    -90deg,
    reflow: true,
    {
      show table.cell.where(x: 1): cell => {
        show regex("\b.+?\b"): it => it.text.codepoints().join(sym.zws)
        cell
      }
      table(
        columns: (3cm, 6.8cm, 7cm, 2.2cm, 3cm),
        table.header(
          [Category],
          [Layer ID#footnote("This is the identifier defined in the geographic data catalogue of Switzerland, geocat.ch.")],
          [Description],
          [Unit],
          [Discretization],
        ),
        [*Needs*],
        [ch.bfe.fernwaerme-nachfrage_industrie],
        [Heating and cooling demand from industry],
        [MWh/year],
        [100m x 100m],

        [],
        [ch.bfe.fernwaerme-nachfrage_wohn_dienstleistungsgebaeude],
        [Heating and cooling demand from residential and commercial buildings],
        [MWh/year],
        [100m x 100m],

        [],
        [ch.are.wohnungsinventar-zweitwohnungsanteil],
        [Electricity needs (estimated from number of residents)],
        [GWh/year],
        [Per municipality],

        [], [ch.bafu.klima-co2_ausstoss_gebaeude], [Heating energy sources in buildings], [-], [Per building],

        [*Potential*],
        [ch.bfe.kleinwasserkraftpotentiale],
        [Potential power of small hydropower plants],
        [kW/m],
        [Per watercourse],

        [],
        [ch.bfe.waermepotential-gewaesser],
        [Potential energy (heat use) of water bodies],
        [GWh/year],
        [Per water body],

        [], [ch.bfe.solarenergie-eignung-daecher], [Potential solar energy from roofs], [kWh/year], [Per roof pane],

        [], [ch.bfe.solarenergie-eignung-fassaden], [Potential solar energy from facades], [kWh/year], [Per facade],

        [], [ch.bfe.biomasse-nicht-verholzt], [Potential energy from biomass], [TJ], [Per municipality],

        [],
        [ch.bfe.fernwaerme-angebot],
        [Potential heat recovery from wastewater treatment plants],
        [MWh/year],
        [Per plant],

        [*Infrastructure*],
        [ch.bfe.statistik-wasserkraftanlagen],
        [Energy from hydropower plants],
        [GWh/year],
        [Per plant],

        [], [ch.bfe.windenergieanlagen], [Energy from wind plants], [GWh/year], [Per turbine],

        [], [ch.bfe.biogasanlagen], [Energy from biogas plants], [kWh/year], [Per plant],
        [], [ch.bfe.kehrichtverbrennungsanlagen], [Energy from waste incineration plants], [MWh/year], [Per plant],

        [],
        [ch.bfe.elektrizitaetsproduktionsanlagen],
        [Energy from electricity production plants (photovoltaic, biomass, geothermal)],
        [kW],
        [Per plant],

        [], [ch.bfe.thermische-netze], [Deliverable energy from thermal networks], [MWh/year], [Per network],
      )
    },
  ),
  caption: "Table of Swiss federal geospatial datasets used by the solution for municipal energy planning, categorized by energy needs, energy potential and existing infrastructure. Each dataset includes the official Swiss geocat.ch identifier, energy units and spatial discretization level.",
) <datasets_table>

==== Guidelines Retriever <guidelines_retriever>

The sole difference between enumerating the data, as collected in the geocontext retriever and proper energy planning lies in the measures that are taken in response to identified issues. Those measures are conditioned by guidelines, broken down into multiple sources.

The primary document called _Vision 2060 et objectifs 2035_ #ref(<StrategieEnergetiqueEnergie>) has been adopted in 2019 and sets intermediate targets for 2035 that take into account the energetical landscape of Valais/Wallis, current knowledge, as well as federal energy and climate policies with the ultimate goal of achieving a 100% renewable and indigenous energy supply in 2060.

Moreover, the _Plan directeur 2019_ #ref(<PlanDirecteurCantonala>) adopted by the federal council on the 1st of May 2019, states the strategy for the canton's territorial development in the form of 49 information sheets, distributed across the five activity sectors: (1) _Agriculture, forest, landscape and nature_, (2) _Tourism and leisure_, (3) _Urbanization_, (4) _Mobility and transport infrastructure_ and (5) _Supply and other infrastructure_.

Finally, the legal framework is defined by two key legislative documents. Notably, the _RS 705.1 - Loi sur les constructions (LC)_ #ref(<RS7051Loia>) establishes the regulations for construction activities, while the _RS 730.1 - Loi sur l'énergie (LcEne)_ #ref(<RS7301Loia>) defines the objectives and requirements for sustainable energy supply.

===== Preprocessing

These documents are specifically designed and structured to convey information to the public and come in a single #acr("PDF") and are available in both french and german. They are organized into sections, subsections or paragraphs which reference figures, tables, plots, past paragraphs and so on.

Visual structure does not necessarily imply a logical flow of information. A document can look and feel organized but still lack a proper machine readable structure.
In practice, it is neither realistic nor scalable to expect a human to manually extract all the key information needed for energy planning from such complex documents. Therefore, it becomes essential to delegate this task to the computer, enabling automated extraction and processing of documents.

When data lack clear structure, it becomes difficult to extract information using algorithms or systematic procedures. However, advances in #acrpl("MLLM") offer a solution as these models are designed to process and understand information presented in various modalities such as text, images, audio and video.

Paired with existing methods that are able to extract raw text from these documents, it has become easier to extract precise information from visually organized and heterogeneous documents by understanding not only the way information is displayed but also its underlying semantic meaning.

As such, a systematic approach is applied when extracting information from these documents:
- Raw text is extracted from the documents on a per-page basis.
- Each page is rendered into an image.
- Each rendered page and associated text are processed using MLLMs to retrieve key insights and interpret the information within the page (#ref(<guidelines_retriever_data_extraction>)).

Qwen2.5-VL is a vision-language language model (multimodal) developed by Alibaba Cloud#footnote("https://www.alibabacloud.com/"), particularly well-suited for processing these complex documents.

Introduced in March 2025, the model demonstrates strong performance, competing that of leading proprietary models in document parsing, structured data extraction and visual reasoning #ref(<baiQwen25VLTechnicalReport2025>).
Consequently, it is able to interpret and analyze figures, plots, tables, etc.

In addition, Qwen2.5-VL is completely open-source and available in compact versions (under 7 billion parameters), making it a great fit for the computational resources that are available.

The extracted information is then formatted in markdown, utilizing headings to structure the summary. It is then broken down into smaller chunks, each chunk being a "chapter" derived from the markdown content.

Since only individual chunks are considered in subsequent steps, there is no need to perform an analysis across neighboring pages to ensure that the information is retrieved from its full context.
These chunks are also translated into english, during the preprocessing step, by the language model.

Finally, the extracted information from each page is encoded into an embedding and stored in the local database, along with its associated chunks and metadata.

Generating embeddings also comes with its own costs, requiring specialized models capable of producing vector representations from text.
As a result, the model nomic-embed-text, developed by Nomic AI#footnote("https://www.nomic.ai/"), is used for used high-performance english text embeddings #ref(<nussbaumNomicEmbedTraining2025>).

It is capable of handling large context windows (the number of tokens that can be processed at once by the model, here 2000), enhancing its ability to capture semantic meaning and improve the quality of information retrieval.
On top of that, it is open-source and available in Ollama, facilitating local deployment.

With clear guidelines now extracted from documents of any format, it is necessary to identify those that are relevant to the user's query. Since those are already embedded, the related guidelines are simply those that are closest to the embedded request in the vector space, as described in the #ref(<geocontext_retriever>) is applied.

Preprocessing these documents and storing them in a database enables the RAG pattern to be leveraged as the agent may now retrieve the relevant information at query time to ground and align its responses with guidelines.

===== Guidelines Interpretation

An issue still lies in how the guidelines themselves are _designed_. While the objectives and figures outlined inside these documents concern the entire canton, this solution is only scoped to municipalities.

Quantitative targets are typically described as such:
#set quote(block: true)
#quote(attribution: [_Vision 2060 et objectifs 2035_ #ref(<StrategieEnergetiqueEnergie>, supplement: [Page 7])])[Ces objectifs de consommation sont multipliés par le nombre d’habitants pour obtenir la consommation pour  l’ensemble du canton, sans les besoins des grands sites industriels. La consommation d’énergie finale pourrait rester stable jusqu’en 2020 (7’960 GWh/a), puis diminuer de 23 % jusqu’en 2035 pour atteindre 6’095 GWh/a. La consommation d’énergie fossile sera amenée à diminuer drastiquement.]
Therefore, they must be scaled down to reflect the municipality’s specific context and expectations.

Identifying which ones require rescaling is a challenging task, as it demands a comprehensive understanding of the broader context.
In order to achieve this, a language model is prompted with the task of identifying key figures that need rescaling (#ref(<guidelines_retriever_system_prompt>)).

Finally, they are multiplied by a factor corresponding to the ratio of the municipality's number of residents to the total population of the canton.
While this is a straightforward way to scale targets, proper rescaling should take into account the economic activity, energy landscape and industrial presence in the municipality to ensure a more accurate adjustment.

The adjusted guidelines are accumulated onto the _context_constraints_ field in the conversational state (#ref(<conversational_state>)).
Similarly to the geospatial information described in the #ref(<geocontext_retriever>), the processed guidelines are accumulated in the state as the conversation goes on and only cleared when switching to a new municipality.

These interfaces are shown in the #ref(<guidelines_retriever_design>).
#figure(
  image("figs/guidelines_retriever_design.svg", width: 12cm),
  caption: "Interface diagram for the Guidelines Retriever agent showing the input conversational context and output as updated state with the retrieved guidelines. The diagram also depicts interaction with the local database storing the embedded regulation documents.",
)<guidelines_retriever_design>

With the relevant guidelines retrieved and rescaled, the query is routed to the strategy planner agent (#ref(<ai_agent_design>), transition 7) which will use them as clear constraints.

==== Municipal Citizen Profile

Before delving into the strategy planner agent, it is necessary to assess how the solution could be enhanced by incorporating additional municipal data.
This segment is deliberately included in the #ref(<system_design>), as this aspect of the solution was envisioned from the start and a proof of concept has been developed.

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

As described in the #ref(<guidelines_retriever>), MLLMs facilitate the extraction of information from these documents.
The only difference lies in how those models are leveraged.

Instead of inquiring the model to summarize or retrieve key insight, a simple citizen profile that is valuable for energy planning is defined#footnote("This profile was defined with the help of the supervisors and could easily be extended.") and searched for, inside each page of the citizen's record (#ref(<municipal_citizen_profile>)).
- The parcel number
- The energy reference surface of the construction, in square meters
- The type of heating system
- The annual energy consumption, in kilowatt-hours per annum
- The power of the solar photovoltaic system, in kilowatt-peak

In the end, the individual pages results are aggregated into a single resident energy profile.

This procedure was tested against a typical citizen record, provided by Prof. Jessen Page. The #ref(<citizen_profile_example>) provides the anonymized (without _parcel_number_), resulting profile.

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
  caption: "Example JSON structure of a Municipal Citizen Profile with building-specific energy data, including parcel number, surface area (m²), heating consumption (kWh/year), heating system type and installed photovoltaic capacity (kWp).",
) <citizen_profile_example>

In a production scenario, these profiles would be generated automatically by setting up data pipelines and stored in a database for further use.

This implementation confirms the feasibility and systematic approach to extracting information from municipal records, supporting the broader vision of the solution's ability to leverage citizen data and gain further insight from these non-public sources.

With this proof of concept demonstrated, the next step covers the design and implementation of the strategy planner agent.

==== Strategy Planner <strategy_planner>

At this stage, every bit of information that is needed to establish a proper energy planning strategy is gathered into the conversational context (#ref(<conversational_state>)).
The prompt of the user has been broken down and analyzed with relevant data points and guidelines retrieved and processed.

In the #ref(<intent_router>), the _intent_ field is defined to either be factual (requesting data) or actionable (seeking planning guidance, recommendations, or strategic advice).
This distinction is crucial as it allows the agent to differentiate between the two tasks, the latter being more expensive in computational resources because of the extra complexity of correlating guidelines and data to concretize a strategy.

Factual queries still contribute to the final goal of establishing an energy planning strategy as it enables users to assess the profile of the municipality and subsequently refine and guide the system into a more effective and informed strategy.

Furthermore, the local database stores user feedback and corrections to past queries.
The agent retrieves pertinent preferences and memories related to the current query and shapes the response according to those expectations.
Like tools and guidelines, memories are stored as embeddings and are therefore retrieved based on semantic similarity.

Finally, similar tools to those retrieved by the geocontext retriever agent are retrieved in order to generate tailored recommendations.
The selection of similar tools is based on their categorization, as defined in #ref(<datasets_table>). This encourages assessing the full spectrum of available data for any municipality.

The state interfaces are presented in the #ref(<strategy_planner_design>).
#figure(
  image("figs/strategy_planner_design.svg", width: 11cm),
  caption: "Interface diagram for the Strategy Planner agent showing the conversational context in input and streamed response in output. The diagram also depicts interaction with the local database, storing the user preferences.",
)<strategy_planner_design>

The factual queries are treated by both #ref(<generate_answer_factual_system_prompt>) and #ref(<generate_answer_factual_user_prompt>) whereas actionable queries are handled by #ref(<generate_answer_actionable_system_prompt>) and #ref(<generate_answer_actionable_user_prompt>).
The conversational context is included in the prompts.

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
  image("figs/critic_design.svg", width: 12cm),
  caption: "Interface diagram of the Critic agent, illustrating in input the generated response and full conversational context. It outputs a boolean decision and can trigger up to three retries if quality criteria are not met.",
)<critic_design>
Its output is a boolean value (#ref(<critic_design>), _retry_) that indicates whether the response has been interpreted correctly based on the rules.

If it the response is not satisfactory, the complete process is restarted as if the user had just prompted the system (#ref(<ai_agent_design>), transition 10).
A maximum of three attempts are allowed before the workflow is not restarted anymore.

At this point, the user's request has been answered and the system is ready to receive a new request, refining the proposed energy planning.

This concludes the design and development of the AI solution. Together, the different components form a robust, multi-agent system that breaks down complex energy planning into smaller, more accessible tasks.

The following section details the implementation choices for the web interface.

=== Web Interface

Designing and providing an interface that is user-friendly and convenient for the user to interact with the AI system is key to the adoption of the product.
The exactitude and accuracy of the AI system weighs heavily on the user's satisfaction but so does the user experience and presentation that is offered.

A web interface is developed to provide a seamless experience for users, offering greater accessibility for devices such as smartphones.

On the development side, the interface was implemented using React#footnote("https://react.dev/"), a popular framework released by Facebook (now Meta) in 2013. React offers a declarative and efficient way to build user interfaces, offering a clean and modular approach using components.

In reality, the choice of framework is not particularly critical in this context. Dozens of frameworks claim to revolutionize the way developers build web applications, but all lead to similar outcomes despite different approaches and philosophies. Past experiences with React and its ecosystem made it a comfortable and efficient choice for this project.

The primary feature of the web interface is the ability to send a prompt to the AI system and have the response streamed back, in real time.
This is achieved using #acr("SSE"), a one-way communication protocol where the server (here the AI agent) pushes events to the client.
In addition to streaming tokens as events, continuous status updates are emitted from the agents to provide live feedback about the progress of the AI solution.

The application being event-driven, SSE was chosen over classic polling mechanisms. Polling requires repeated requests from the client to the server, which can increase both server load and response latency whereas SSE maintains a single persistent connection on which events are pushed.

On top of that, the use-case of energy planning for municipalities greatly benefits from a map, displaying the assessed data points.
This functionality recovers a problem covered in the #ref(<geocontext_retriever>): the loss of information due to aggregation.

All datasets referenced in #ref(<datasets_table>) provide layers, along with their discretized features. Interpreting them visually preserves the local variation and allows for a more nuanced understanding of the reported aggregated values.

These data points and their sources are also presented in the interface, allowing users to assess and verify the accuracy of the reported energy planning. This transparency is offered to further build user trust in the solution.

On the implementation side, OpenLayers#footnote("https://openlayers.org/") is a robust and flexible open-source library for building interactive web maps.
Native support for #acr("WMTS") enables the integration of tiled map services such as those provided by GeoAdmin. These tiles are high-resolution images enabling efficient and scalable map rendering by only loading visible portions of the map. GeoAdmin notably makes use of OpenLayers in their services.

With the ultimate goal of assisting users in energy planning, downplaying the importance of durability aspects of the solution would be a significant oversight.
AI progress is often celebrated, yet the energetical impact of these systems is frequently overlooked.
While a complete evaluation of the energetical footprint of the solution is clearly beyond this work's scope, a simple approach has been implemented to sensitize users to the matter.

For each user prompt, the cumulated number of tokens that are input and output from the agents in #ref(<ai_agent_design>) is recorded.

The cumulative count of prompts and average token count per prompt are incrementally calculated and stored in the database.

Along that, Welford's online algorithm allows for the calculation of the variance, in a single pass (#ref(<welford>)).
It defines a recurrence relation for updating the sum of squared differences from the current mean, allowing to compute the variance incrementally.

This algorithm is numerically stable and does not require storing all the data points, reducing the memory footprint of the system.

To sum up, the token utilization of users is tracked in the form of three metrics only: (1) the average token count per prompt, (2) the sum of squared differences from the current mean, from which the variance can be computed and (3) the cumulative count of prompts.

When users prompt the AI, the cumulative token count for that run is monitored. This value is then compared against the user's sampled token utilization distribution using the standard score (#ref(<zscore>)), a statistical measure that expresses how many standard deviations away a value is from the mean, assessing how far the new usage deviates from the user's average.

Accordingly, the token usage of the current prompt is categorized into one of the predefined categories: "bad", "average", or "good"; each associated with a color pelet displayed in the interface.

Moreover, an energy consumption analogy is presented alongside the pelet.
The research article _Beyond Test-Time Compute Strategies: Advocating Energy-per-Token in LLM Inference_ #ref(<wilhelmTestTimeComputeStrategies2025>) provides benchmarks indicating that language models of similar size, on average, consume 3 Joules of energy, per token, during inference.

As such, the energy consumption of the current prompt is estimated using the simple heuristic of 3 Joules per token.
It is then expressed as the equivalent duration, in minutes, a 10 watt LED light bulb could run with the same amount of energy (#ref(<token_consumption>)).

This implementation is strictly meant to raise awareness about the energy consumption associated with AI usage.
In this solution, requesting factual data from the system is more energy-efficient than inquiring actionable planning.
This is because factual queries involve fewer agents in the workflow defined in #ref(<ai_agent_design>).

With that, the implementation details of the web interface are clarified. This highlights its role as being on par with the AI agent solution itself.

== Limitations <limitations>

Every technical solution, regardless of how well designed and implemented, is subject to different limitations.
These constraints often stem from underlying assumptions made during the design process or specific implementation details.

Identifying them is a first step towards a self-evaluation of the solution. Please note that the limitations outlined in this section are not exhaustive.

When integrating external services and data sources, it is very difficult to assess the quality and exactitude of the data.
Inaccuracies lead to incorrect interpretations and conclusions reported by the system, degrading the performance and reliability of the system.

The various offices referenced in the #ref(<geocontext_retriever>), commissioned to retrieve and publish data, address these issues by implementing different quality assessment procedures.

The SFOE, for example, only tolerates a small proportion of errors and gaps in the data they collect#footnote("https://www.bfs.admin.ch/").

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

This is shown in the paper _Emergent Abilities of Large Language Models_ #ref(<weiEmergentAbilitiesLarge2022>), published in the _Transactions of Machine Learning Research_, in 2022.
These capabilities are relevant for the strategy planner agent which must synthesize large context to deduce energy planning insights and recommendations.

To support this, greater models run on _Calypso_, a sandbox infrastructure designed and reserved for students of the bachelor program.
Nevertheless, the largest model that fits on this infrastructure (around 8 billion parameters) remains relatively small compared to state-of-the-art large language models (>150 billion parameters).
Exploring the use of -really- large language models in future implementations could lead to improvements in interpretation and overall quality of the system.

Furthermore, a strong limitation of the solution is the inability of the user to provide and induce bias in the system#footnote("While no extensive jailbreaking or bias testing was conducted, all attempts made during the development process to introduce bias into the system were unsuccessful.").

This might be seen as a good thing, as it guarantees consistency in the output but this also means that the system is less flexible and adaptable to the individual preferences, explained in the #ref(<intent_router>).

While this ensures consistent output, it also reduces the flexibility and adaptability of the system to put into service all user preferences.
These instructions are distinguished into two categories: presentation directives (1) and effective instructions (2).

Presentation directives define how the data shall be formatted (table, bullet points, etc.) and which aspects of the data shall be prioritized or highlighted. This structures the response and emphasizes specific details that are key to the users.

Effective instructions, on the other hand, may attempt to substitute the official data sources retrieved by the system with user-provided knowledge or simply refine and guide the output of the system to better align with user needs.

Informed users may provide more accurate and precise figures regarding the datasets referenced in #ref(<datasets_table>) but their contributions are completely ignored as the solution prioritizes its own retrieved data.
An example of this type of interaction can be found in #ref(<fake_data>).

The system is guided by the different prompts to strictly operate based on the workflow defined in #ref(<ai_agent_design>) and offer assistance in energy planning tasks.
Providing more detailed instructions in the prompt may enable this behavior.

Leveraging them broadens the scope of tasks that can be handled by the agents. General use cases for these agents include data processing and analysis, interaction with APIs and arithmetics.

The retrieval, processing and aggregation of the data sources presented in the #ref(<geocontext_retriever>) would be an excellent application of their capabilities.
Rather than relying on static, predefined procedures for data retrieval, agents could dynamically generate and execute code to access and process data from larger and more diverse datasets, naturally expanding the range of available data sources.

The classic agent architecture adopted in this work ensures a more transparent and traceable workflow where the role and interface of each agent is clearly defined.
While this design choice facilitates maintainability and allows for a more reliable and structured handling of complex geospatial data, it would still be interesting to explore the paradigm shift towards coding agents.

Outside of the data retrieval and processing, these agents would greatly enhance the critic of the system's response (#ref(<critic>)).
A typical energy profile, including relevant indicators from the datasets in #ref(<datasets_table>) could be pre-defined and, for example, standardized to a per-resident basis.

By doing so, the agent could evaluate the accuracy of the values presented in the energy planning report, converting them to the same reference scale and comparing them against the average profile.
This would enable the identification of subtle inconsistencies, leading to a more robust and reliable implementation.

In summary, these points illustrate some of the current limitations inherent to the system. The following chapter presents the results obtained from the implemented solution.

= Results <results>

This chapter presents the empirical findings from the assessment of the implemented system.
A structured testing methodology is established, facilitating the evaluation of the solution's primary objective: assisting users in municipal energy planning.

== Evaluation Methodology

To begin, it is essential to define the methodology and criteria used to evaluate the system's performance.

There is no definitive ground truth in energy planning, making it difficult to quantify the accuracy of the reported recommendations and strategies.
Consequently, qualitative observations provide valuable insight into the practical effectiveness of the solution.

These insights are provided by two sources:
- An expert assessment, provided by Prof. Jessen Page
- An automated evaluation using a LLM-as-a-judge benchmarking framework

The latter leverages language models to mimic expert judgment, assessing the response against a set of predefined criteria.
This approach provides a scalable and consistent alternative to human evaluation.

=== LLM-as-a-judge Benchmarking Framework
Unlike human experts which may emphasize different aspects depending on their interpretation or even mood, language model evaluation is driven by clear rules, ensuring consistency and uniformity across all cases. The G-Eval metric, as introduced in the #ref(<state_of_art>), summarizes the score of the criteria and quantifies the quality of the response.

The criteria for the LLM-as-a-judge benchmarking framework are defined as follows:
1. Data interpretation: assesses whether the response uses only relevant data, maintains mathematical accuracy, distinguishes energy types, preserves units and handles zero values correctly.
2. Methodology alignment:
  - For factual queries: checks clear data analysis, insight identification and avoidance of proper plannification.
  - For actionable queries: evaluates structured planning, guidelines integration and expert positioning.
3. Municipal relevance: rates feasibility at local scale, direct query alignment, consideration of local context and actionable next steps.
4. Technical compliance: checks language consistency, correct structure, citation format and completeness of required sections.

And evaluated according to the following scale, presented in #ref(<evaluation_grid>).
#set table(
  fill: (x, y) => { if calc.odd(y) { rgb("D8E1E6") } },
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
  caption: "Five-point ordinal evaluation scale used in the LLM-as-a-judge benchmarking framework to assess AI agent responses across multiple criteria. Each score represents a quality level from (1) fundamentally flawed to (5) exceptional performance.",
) <evaluation_grid>

These criteria are based on the most frequent types of errors, encountered during the continuous assessment of the solution.

These instructions are summarized and included in the #ref(<benchmark_system_prompt>). The conversational context is also included, helping the model to evaluate the accuracy of the generated response against the retrieved data points and guidelines.

Finally, the individual benchmark scores are aggregated into a single score by calculating the arithmetic mean and then rescaled linearly to the [0,1] interval.
This yields an easily interpretable overall performance metric, better known as G-eval.

=== Expert Assessment Framework

Human insight is naturally shaped by the domain knowledge and nuanced judgment of the expert, leading to a more informed assessment.

While the automated evaluation is constrained to a rigid scoring grid, the expert evaluation is richer as observations extend beyond these predefined criteria and reflects interpreted, context-specific priorities.
As such, it is not feasible to enforce strict scoring rules to the expert.

However, the grid presented in #ref(<scoring_grid_expert>) acts as a reference point and guides the nuanced and _unlimited_ qualitative feedback to a single quantitative score, allowing for further comparison and analysis.
#figure(
  table(
    columns: 3,
    table.header([Score], [Label], [Description]),
    [1],
    [Not relevant],
    [Information is completely off-topic; does not answer the question, is generic or incoherent in the context of energy planning.],

    [2],
    [Weakly relevant],
    [Some elements are related to the subject; the response is mostly vague, imprecise, or off-topic. It could mislead an expert.],

    [3],
    [Moderately relevant],
    [Response is generally on theme but remains partial, imprecise, or incomplete; requires significant corrections to be useful.],

    [4],
    [Relevant],
    [Information is correct, targeted and generally adapted to the question; minor adjustments may be needed but it is usable for planning.],

    [5],
    [Highly relevant],
    [Information is perfectly aligned with the question, complete and contextualized; no corrections are necessary and it is ready to be used as-is for decision-making.],
  ),
  caption: "Five-point ordinal evaluation scale for expert assessment of AI-generated energy planning responses, ranging from completely irrelevant (1) to highly relevant and ready-to-use (5).",
) <scoring_grid_expert>

=== Framework Interpretability

It is important to note that the G-eval and expert scores cannot be directly compared, each being grounded in a distinct evaluation framework.

G-eval offers a standardized framework with set evaluation criteria, allowing for a more consistent and reliable benchmarking. This enables the comparison of different solutions under identical conditions.

As G-eval relies on language models, scores may showcase some degree of randomness. This is mitigated by multiple runs and aggregated scores, offering a stable estimate.

By presenting both evaluation methods, the objectivity of an automated scoring is complemented by the more practice-oriented expert judgment.
This dual approach treats both methodological rigor and contextual relevance to assess the quality of the solution.

For consistency and easier interpretation, expert scores are also rescaled linearly to the [0, 1] interval.

#pagebreak()
=== Test Dataset

With the evaluation frameworks introduced, the next step is to define the test dataset.
This dataset consists of nine prompts and establishes the basis for assessing the performance of the solution (#ref(<test_dataset>)).
#set table(
  fill: (x, y) => if calc.odd(y) { rgb("D8E1E6") },
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
  caption: "Comprehensive test dataset of nine evaluation prompts designed for assessing the peformance of the system in municipal energy planning for Sion. This dataset enables systematic evaluation of both factual information retrieval and actionable planning recommendation capabilities.",
) <test_dataset>

The dataset is aligned within the specific scope of available data sources, inherently restricting its size.

It is crafted by Prof. Jessen Page to support energy planning for the municipality of Sion.
The responses generated by the solution to each question are presentend in #ref(<evaluation_data>).

With the groundwork established, the following section presents the results that will be discussed in the next chapter.

Both the expert assessment and benchmarking are conducted using the version of the solution as of June 20, 2025#footnote[Code at commit `c2bea64` in the repository and raw results in `evaluation` branch.], ensuring consistency and reliability in the evaluation process and reported conclusions.

== Evaluation Results

Two different configurations of the solution, each using different language models, are benchmarked.
The #ref(<configurations>) breaks down their composition.

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
    [*Small*], [qwen3:1.7B], [qwen3:1.7B], [qwen3:1.7B], [qwen3:1.7B], [deepseek-r1:8B],
    [*Large (Baseline)*], [qwen3:1.7B], [qwen3:1.7B], [qwen3:1.7B], [*qwen3:8B*], [deepseek-r1:8B],
  ),
  caption: "Language model choices across different agent configurations used in the evaluation study. The small configuration uses lightweight 1.7B parameter Qwen3 models for all agents, while the large baseline configuration make use of a more powerful 8B parameter Qwen3 model specifically for the Strategy Planner agent, which handles the most complex reasoning tasks. The DeepSeek R1 8B model serves as the evaluator for G-eval benchmarking across both configurations to minimize bias in assessment.",
) <configurations>

A smaller configuration is defined for further comparison against the baseline.
This baseline, a _larger_ configuration, leverages bigger language models for the strategy planner agent as introduced in the #ref(<limitations>).

The general-purpose Qwen series offers tool-using abilities, reasoning and model size, making it the only currently available option in Ollama that is viable for running lightweight, yet capable agents.
Deepseek models, on the other hand, are developed by Deepseek, a company funded by the High-Flyer#footnote("https://www.high-flyer.cn/") hedge fund and whose models support similar capabilities to Qwen. They are also open-source.

The R1 model (8 billion parameters) serves as the evaluator for all LLM-as-a-judge benchmarking. Using a model from a different "family" than those in the individual agents helps minimize potential bias in the assessment.

It is also the only other model available in Ollama that offers both reasoning and tool-using capabilities within the available computational resources.

The #ref(<comparison_test_human_llm>) presents the results, showing both the expert assessment and G-eval benchmarking scores for all nine prompts, side by side and for both configurations.
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
    columns: (2.5cm, 4cm, 4cm, 4cm),
    align: center,
    table.header(
      table.cell(rowspan: 2, []),
      table.cell(rowspan: 2, []),
      table.cell(rowspan: 2, colspan: 2, [G-eval (mean ± st.d.) [-], 10 runs]),
    ),
    [Prompt No°], [*Expert score*], [Small configuration], [*Large configuration*],
    [1], [0.50], [0.23 ± 0.08], [0.39 ± 0.15],
    [2], [0.25], [0.26 ± 0.10], [0.43 ± 0.11],
    [3], [0.50], [0.36 ± 0.06], [0.44 ± 0.11],
    [4], [0.25], [0.23 ± 0.08], [0.37 ± 0.22],
    [5], [0.75], [0.38 ± 0.00], [0.39 ± 0.17],
    [6], [0.50], [0.23 ± 0.08], [0.41 ± 0.15],
    [7], [0.75], [0.28 ± 0.10], [0.44 ± 0.10],
    [8], [0.25], [0.38 ± 0.00], [0.49 ± 0.13],
    [9], [0.25], [0.21 ± 0.06], [0.50 ± 0.11],
  ),
  caption: "Comparative evaluation results showing expert assessment scores (normalized 0-1 scale) versus G-eval automated benchmarking scores (mean ± standard deviation over 10 runs) for both system configurations across all nine test prompts. The table demonstrates the performance difference between small and large configurations, with the large configuration consistently outperforming the small one in automated evaluation.",
) <comparison_test_human_llm>

While both frameworks are not directly comparable, this provides an interesting perspective for further analysis.

On the other hand, the scores per benchmarking criterion and per query _intent_ type (either factual or actionable, per definition), for both configurations, are depicted in the #ref(<boxplots_criterion>).
#figure(
  image("figs/boxplots_criterion_intent.png", width: 100%),
  caption: "Box plot analysis showing the distribution of G-eval benchmark scores across the four evaluation criteria (Data Interpretation, Methodology Alignment, Municipal Relevance, Technical Compliance) grouped by query intent type (Factual vs. Actionable) for both small and large system configurations. The plots reveal performance variations between criteria and configurations.",
)<boxplots_criterion>

The mean score and entropy of the scores distribution, per benchmarking criterion and per query _intent_ type are reported in the #ref(<comparison_criteria>) and #ref(<entropy_criteria>).

#set table(
  stroke: (x, y) => {
    if y == 4 {
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
    if x == 4 and y == 4 {
      (right: 0.3pt + black)
    }
  },
  fill: (x, y) => {
    if y < 4 and x == 0 { none } else {
      if calc.odd(y) { rgb("D8E1E6") }
    }
  },
  align: (x, _) => if x == 0 { left } else { center },
)

#figure(
  table(
    columns: 5,
    table.header(
      table.cell(rowspan: 3, []),
      table.cell(rowspan: 3, colspan: 4, [Mean score [-], 10 runs]),
    ),
    [], table.cell(colspan: 2, [Factual]), table.cell(colspan: 2, [Actionable]),
    [*Criterion*], [Small], [*Large*], [Small], [*Large*],
    [Data Interpretation], [1.45], [2.55], [1.70], [2.74],
    [Methodology Alignment],
    [1.55],
    [2.20],
    [2.19],
    [3.07],
    [Municipal Relevance],
    [3.00],
    [3.50],
    [2.54],
    [2.93],
    [Technical Compliance], [1.60], [1.80], [2.33], [2.34],
  ),
  caption: "Mean performance scores from five-point ordinal evaluation across four evaluation criteria, by query intent type (factual and actionable) and system configuration. The results show the superior performance of the large configuration and global challenges in Data interpretation and Technical compliance.",
) <comparison_criteria>

#figure(
  table(
    columns: 5,
    table.header(
      table.cell(rowspan: 3, []),
      table.cell(rowspan: 3, colspan: 4, [Entropy [nats], 10 runs]),
    ),
    [], table.cell(colspan: 2, [Factual]), table.cell(colspan: 2, [Actionable]),
    [*Criterion*], [Small], [*Large*], [Small], [*Large*],
    [Data interpretation], [0.69], [1.19], [0.69], [1.38],
    [Methodology Alignment],
    [0.69],
    [1.25],
    [1.14],
    [1.54],
    [Municipal Relevance],
    [0.00],
    [1.11],
    [1.00],
    [1.50],
    [Source Citations], [0.50], [0.95], [0.69], [1.16],
  ),
  caption: "Entropy analysis (in nats) of G-eval score distributions across evaluation criteria and query types, measuring the variability and consistency of system performance. The large configuration shows generally higher entropy, particularly for actionable queries, indicating more varied responses.",
) <entropy_criteria>

It is important to note that the distribution of prompts by intent is unbalanced as only two prompts from the test dataset (#ref(<test_dataset>)) are classified as "factual" (no° 1 and 4), while the seven remaining ones are classified as "actionable".

These results summarize and introduce further comparison between the expert evaluation and G-eval benchmarking framework, assessing differences between the baseline configuration and a smaller one.
With the different results presented, the following chapter discusses their implications.

= Discussion

This chapter addresses the results presented in the #ref(<results>) and connects them to the research question: evaluating the effectiveness and reliability of AI agents for urban energy planning, along with an analysis of their strengths and weaknesses.

== Evaluation Paradigms <eval_paradigm>

The #ref(<results>) differentiates the two evaluation frameworks: (1) the expert assessment framework and (2) the G-eval benchmarking framework.

As both rely on different methods and criteria, they provide complementary insights rather than direct comparability.
The expert evaluation provides nuanced and domain-informed feedback while the G-eval framework delivers a standardized and consistent assessment based on predefined criteria.

Both frameworks and their scores are presented in the #ref(<comparison_test_human_llm>), side-by-side against each prompt of the test dataset.

The relationship between these two frameworks is assessed by leveraging Spearman's rank correlation coefficient (#ref(<spearman>)).

This coefficient measures the monotonic relationship between the rankings of two variables, indicating how well the order of one variable matches the order of the other. Like other correlation coefficients, it varies between -1 (perfect inverse correlation) and +1 (perfect correlation), with 0 implying no correlation at all.

This coefficient strictly applies to ordinal data. As both score distributions originate from a ranking, the coefficient is applicable to compare the relative ranking of prompts of the two frameworks.

As such, the calculated Spearman correlation coefficient between the expert evaluation and the G-eval framework is -0.23 for the larger configuration and 0.36 for the smaller configuration, indicating no meaningful correlation between the two frameworks.

Despite the smaller configuration showing a weak trend, it is not statistically significant as the sample size is too small.
This divergence demonstrates that the G-eval framework captures distinct dimensions than those of the expert evaluation, validating the complementary nature of the evaluation methods.

Interestingly, the correlations for the larger configuration and smaller configuration, against the expert evaluation, differ significantly.

To further understand this difference, the Spearman correlation between the G-eval scores of the large and small configurations is calculated.
The resulting coefficient of 0.09 suggests that the resulting responses of the two configurations are ranked substantially differently, opening up the possibility of exploring the reasons behind this discrepancy.

Consequently, the G-eval framework is, at most, a complementary tool to the expert evaluation.
With that in mind, the results can now be compared in depth to gain additional insight into the performance of the solution.

== Performance Analysis <perf>

On top of that, the same #ref(<comparison_test_human_llm>) depicts the average G-eval scores for both configurations, per prompt. Visually, the larger configuration consistently outperforms the smaller configuration.

This can be formally assessed by conducting a Wilcoxon signed-rank test, whose null hypothesis states that two randomly selected samples from two populations have the same distribution.
Besides that, it is non-parametric, making no assumptions about the underlying distribution, making it more robust to outliers.
#highlight("TODO: citer wilcoxon")

Therefore, it is applied to the paired scores of the larger and smaller configurations with the one-sided alternative hypothesis that the larger configuration significantly outperforms the smaller one.
The resulting p-value is 0.009 (= 0.9%), indicating a statistically significant difference between the two configurations considering a 5% confidence level.

The null hypothesis is hence rejected and confirms the alternative hypothesis: the larger configuration, indeed, outperforms the smaller configuration in a per-prompt basis and verifies the initial expectation that larger language models, incorporated into the AI agent solution, yield better results.

Analyzing the per-criterion score distribution across prompt intent (#ref(<boxplots_criterion>)) provides valuable insight into the specific areas in which the larger configuration surpasses the smaller one.

Boxplots enable a visual comparison of the distributions of ordinal scores, assigned to each evaluation criterion.
The results are further grouped by prompt intent, distinguishing between factual and actionable prompts, the latter of which involve strategy planning and greater analysis.

The plot clearly demonstrates that the larger configuration is consistently ranked higher across all criteria, raising an important question: in which aspect of the qualitative evaluation does the larger configuration have the greatest impact ?

The mean scores, reported in #ref(<comparison_criteria>) support that the data interpretation (+76% and +61%) and methodology alignment (+42% and +40%) show the most significant improvement, in factual and actionable contexts.
This suggests that bigger models enhance the system's ability to interpret the geospatial data and provide actionable insights from various perspectives, identifying patterns and most importantly suggesting measures.

In contrast, municipal relevance and technical compliance only show slight improvements (<\17%) between configurations.
What is more interesting is the overall performance of the solution, across all configurations for these two specific criteria.

While the municipal relevance criterion reveals the greatest score across all criteria and a more pronounced increase (+17% and +15%) between configurations, the technical compliance criterion remains strikingly similar across both (+13% and +1%).

The technical compliance results are likely due to either: poor compliance with the formatting requirements, structural completeness and overall presentation of the response for both configurations (as defined by the criterion, #ref(<benchmark_system_prompt>)) or a more fundamental problem with the criterion itself.

In reality, both go hand in hand as overly rigid criteria penalize small deviations from the requirements whilst loose criteria may overlook important details and yield higher scores.
On the other hand, the criterion may be irrelevant. A response that effectively addresses urban energy planning requirements, even lacking perfect formatting, is arguably better than one that is well-formatted but completely meaningless.

Another issue that is observed with it is the strong variability in the scores that are assigned, over both query intents.
More generally, high variability is measured throughout all criteria, suggesting a broader issue.

The score distributions in #ref(<boxplots_criterion>) display a large visual spread and clear outliers in criteria such as methodology alignment and technical compliance.
Moreover, this is assessed from the information theory entropy (#ref(<entropy>)), a measure that quantifies the average level of uncertainty around the potential states of a variable, here, every criterion across configurations and prompt intents.

If a criterion is consistently scored the same, then each occurrence of that score is highly predictable and conveys no information (low entropy).
Conversely, when the distribution of scores of a criterion is more diverse, each result is more _surprising_ and therefore carries greater information.

When the scores are distributed uniformly, entropy reaches its maximum, where each score is equally likely. For the 1-to-5 ranking scale used here, the maximum entropy is 1.61 nats (#ref(<entropy_upper_bound>)).

With that in mind, the entropies in #ref(<entropy_criteria>) validate the visual feeling of diversity with almost all criteria reaching absurd levels of uncertainty.
The methodology alignment and municipal relevance for the large configuration, for example, show entropies of respectively 1.54 nats and 1.50 nats, very close to what would be the results of a fair 5-faced dice, nearly fully random.

Globally, the larger configuration shows higher entropy over all criteria, suggesting that its responses vary more widely in their compliance with the evaluation standards.
This volatility reflects stochastic fluctuations across the repeated benchmarks on the same set of prompts, rather than differences caused by prompt diversity.

While evaluating scores across a large and varied set of prompts provides insight into the robustness and generalization of the AI agent, the setup presented here focuses on the consistency and reliability of the benchmarking framework itself, as it is the foundation of more ambitious and extensive automated evaluations.

By always setting the temperature parameter of the language models to zero, the models always select the most probable next word when generating responses. This maximizes the consistency of the results in both benchmark scoring and response generation.

Consequently, this fluctuation may point to (1) a lack of robustness in the benchmarking framework or (2) a statistical artifact of the limited number of benchmark runs.

During this work, only few benchmarks runs were conducted (10) as a result of time and resources constraints.
Therefore, further experimentation through additional runs is needed to distinguish between these possibilities.

If the _intra-prompt_ variability remains, it points out to an issue with the benchmarking framework and its evaluator model, unable to grasp and apply the evaluation criteria with nuance. Models may lack the semantic sensitivity needed to assess complex requirements.
Conversely, if it decreases, it can then be attributed to a statistical irregularity due to few benchmark iterations.

When interpreting the results from a per-intent perspective, a clear trend emerges as the larger configuration outperforms the smaller one on actionable prompts, especially in criteria that involve increased reasoning and multi-step tasks, such as data interpretation and methodology alignment.
The sole criterion that is scored higher in factual contexts is the municipal relevance criterion.

This hints at the ability of the system to perform better in less constrained, strategy-oriented tasks, rather than factual ones.
However, this may also allude to an asymmetry in how the evaluation criteria are interpreted over intent types.

Data interpretation and methodology alignment may be inherently biased towards actionable prompts due to their nature of requiring more nuanced reasoning and context understanding, reinforcing previous concerns regarding the definition of the criteria.

Alongside the qualitative scoring of the expert, annotations were also collected.
#highlight("TODO: donner résultats raw expert + benchmarking?")

The feedback highlights recurring issues observed in the test case, emphasizing certain systematic weaknesses in the solution:
- Unsupported conclusions and takeaways, often delivered with high confidence.
- Mix of municipal data and unscaled cantonal data, leading to inconsistent or misleading figures.
- Misinterpretation of request, such as discussing production when only demand was requested.

While informed users might identify these inconsistencies and filter them out to extract meaningful insights, unfamiliar users may be deceived.
Overall, these observations highlight both the potential and the current limitations of this project, as demonstrated through this analysis.

The implications are further discussed in the #ref(<close>) and most importantly, brought back to the initial research question.

== Research Takeaways <close>

By drawing from the results of both expert assessment and LLM-as-a-judge benchmarking, this section evaluates the ability of the proposed solution to assist users into urban energy planning tasks.
The analysis states the main strengths and weaknesses of the system in terms of (1) effectiveness and (2) reliability.

Contextual reasoning and structured planning are typical tasks, encountered in energy planning. They involve the synthesis of diverse data and context into a strategy that aligns with the municipality and ensures compliance with regulatory frameworks.

The solution shows clear strengths in those areas, more particularly in contexts requiring proper analysis and the establishment of clear strategies.

Furthermore, the larger configuration which leverages more powerful language models achieves higher scores than the smaller one, across all criteria defined in the automated evaluation.
This verifies the core assumption that greater models result in greater quality of responses.

While the system is effective in delivering meaningful and interpretable energy planning guidance, this does not guarantee its reliability.

The reported strategies often present unsupported claims, variable interpretations or ambiguous assumptions; critical flaws that undermine the agent's reliability.
These issues are observed and noted from the ongoing assessment of the solution and comments on the resulting recommendations.

Besides that, the evaluation methodology falls short in the distinction of what may be an architectural issue with the benchmarking framework and its criteria or simply areas of the responses that need improvement.
This effect is further amplified by the stochastic nature of language models, introducing challenges for repeatability and quality assurance.

Although the system lacks the reliability needed for reliable decision-making, it does show assistance capacities.

The retrieval, contextualization and presentation of geospatial data from different perspectives, driven from a simple natural language query, fulfills a key feature that is expected of an assistant, supporting users in typical energy planning tasks.

In contrast, while expert users can effectively retrieve and refine the suggestions of the agent for practical planning, uninformed users are at greater risk of being misled.

Aside of the architectural and implementation-specific limitations presented in #ref(<limitations>), extended testing that includes more benchmark runs and the use of larger, more capable, language models is necessary to validate the assumptions regarding ranking variability across responses to isolate more granular issues.

At this stage, the solution shows promise and supports the potential for AI agents in urban energy planning. However, it remains fragile and requires further development to achieve production-grade robustness.

= Conclusion

In summary, this work investigates the effectiveness and reliability of AI agents for urban energy planning, aiming to assist users into grounded and data-driven decisions at a municipal level.
Therefore, a multi-agent architecture was adopted to decompose the complex task of energy planning into smaller, more manageable subtasks, each handled by a specialized agent.

This chapter summarizes the main findings and contributions within this work and suggests directions for future research and development.

== Main Findings

While the assessment of the solution shows undeniable potential in supporting municipal energy planning tasks, it also highlights important limitations.

In terms of effectiveness, the system successfully addresses energy planning queries by coordinating the different AI agents.
Among its main strengths are its capabilities in contextual reasoning and structured planning tasks for queries that require strategic analysis and synthesis of diverse data sources.

Larger language models demonstrate superior performance across all evaluation criteria, with the most significant improvements in data interpretation (+76% for factual queries, +61% for actionable queries) and methodology alignment (+42% for factual queries, +40% for actionable queries).

Despite its effectiveness, the solution exhibits concerning reliability issues.
The expert assessment identified recurring problems, such as unsupported conclusions that are delivered with a very confident tone or the misinterpretation of data sources leading to misleading figures and analysis.

These issues pose particular risks for uninformed users, which may lack the expertise and knowledge to identify and filter out inconsistencies from the relevant insights.

== Key Contributions

Various contributions are outlined in this section, based on the implementation and assessment of the solution:
1. A modular multi-agent architecture composed of six AI agents, each integrated through a unified conversational workflow, enabling the solution to exploit the capabilities of individual and specialized agents, while maintaining a clear and deterministic flow of information.
2. Publicly available datasets and policy frameworks are incorporated into the system, ensuring that its reasoning is grounded in real-world planning constraints and sustainability requirements, both essential for effective urban energy planning.
3. The evaluation framework combines expert assessment with LLM-as-a-judge automated benchmarking, merging nuanced qualitative feedback with a standardized evaluation based on predefined criteria, enabling large-scale comparison across diverse configurations.

== Future Work

The findings presented in this thesis open up several promising future implementations and investigations, from which a non-exhaustive list is outlined in this section:
- Exploring the integration of large language models into the solution, addressing current limitations in data interpretation and consistency.
- Extending both the capabilities of the automated benchmarking framework, leveraging finer-grained criteria to better identify areas of improvement and the number of runs to assess the inconsistency of the results.
- Integrating coding agents, transitioning from the current static data retrieval and analysis paradigm to fully autonomous data processing, delivering richer insights from broader data sources.
- Developing a specialized classification model for the intent router agent, reducing the computational cost and latency of the solution as well as improving the accuracy of intent recognition.
- Expanding the scope of supported municipalities outside of Valais/Wallis.
- Enhancing the user experience, supporting mobile devices, improving the design and accessibility of the web interface, generating PDF exports, etc.

== Closing Remarks

While the proposed solution marks a step towards smarter data-driven decision support tools, for municipalities, it does also highlight the ongoing need for research into reliability and transparency, enhancing the robustness of the solution.

Ultimately, this thesis demonstrates both the promise and the current challenges of AI agents in the field of urban energy planning.

#pagebreak()
#heavy-title(i18n(doc_language, "bibliography-title"), mult: 1, top: 0.5em, bottom: 0.3em)
// generate bib file RIS script (https://www.bruot.org/ris2bib/)
#bibliography("bibliography.bib", full: true, style: "ieee", title: none)

//////////////
// Appendices
//////////////
#cleardoublepage()
#appendix-page()
#pagebreak()
#heavy-title("A. Prompts", top: 1em, bottom: 1em)

#let prompt_creation = read("code/prompt_creation.txt")
#figure(
  code()[
    #raw(prompt_creation)
  ],
  caption: "Meta-prompt used for systematic prompt engineering",
  kind: "prompt",
  supplement: [Prompt],
) <prompt_creation>

#let intent_router_prompts_system = read("code/intent_router_prompt_system.py")
#figure(
  code()[
    #raw(intent_router_prompts_system, lang: "python")
  ],
  caption: "System prompt for the Intent Router agent that defines its role in analyzing user queries to extract location information and classify intent as either factual (data requests) or actionable (planning recommendations). The prompt establishes the agent's responsibility for parsing natural language queries.",
  kind: "prompt",
  supplement: [Prompt],
) <intent_router_prompt_system>
#pagebreak()
#let intent_router_prompts_user = read("code/intent_router_prompt_user.py")
#figure(
  code()[
    #raw(intent_router_prompts_user, lang: "python")
  ],
  caption: "User prompt template for the Intent Router agent that provides the specific user query and conversation context for intent classification. This template structures the input data including the user's current and past question as well as the context.",
  kind: "prompt",
  supplement: [Prompt],
) <intent_router_prompt_user>

#let clarify_query_prompts_system = read("code/clarify_query_system_prompt.py")
#figure(
  code()[
    #raw(clarify_query_prompts_system, lang: "python")
  ],
  caption: "System prompt for the Clarify Query agent that defines its role in handling ambiguous or incomplete user requests by generating targeted clarification questions. The prompt instructs the agent to identify missing information, formulate clear and specific follow-up questions and maintain a conversational tone while ensuring all necessary details for energy planning analysis are obtained from the user.",
  kind: "prompt",
  supplement: [Prompt],
) <clarify_query_system_prompt>

#let clarify_query_prompts_user = read("code/clarify_query_user_prompt.py")
#figure(
  code()[
    #raw(clarify_query_prompts_user, lang: "python")
  ],
  caption: "User prompt template for the Clarify Query agent that structures requests for clarification by combining the original user query with identified missing information fields. This template enables the agent to generate contextually appropriate clarification questions that help users refine their energy planning inquiries.",
  kind: "prompt",
  supplement: [Prompt],
) <clarify_query_user_prompt>

#let geocontext_retriever_prompts_system = read("code/geocontext_retriever_system_prompt.py")
#figure(
  code()[
    #raw(geocontext_retriever_prompts_system, lang: "python")
  ],
  caption: "System prompt for the Geocontext Retriever agent that establishes its role in selecting and executing appropriate geospatial analysis tools based on user queries and municipal context. The prompt defines how the agent should interpret geographical requirements, choose relevant datasets and format spatial information for downstream processing in the energy planning workflow.",
  kind: "prompt",
  supplement: [Prompt],
) <geocontext_retriever_system_prompt>
#pagebreak()
#let infer_pages_prompts_system = read("code/infer_pages_system_prompt.py")
#figure(
  code()[
    #raw(infer_pages_prompts_system, lang: "python")
  ],
  caption: "System prompt for the data extraction component of the Guidelines Retriever that processes Swiss energy planning and guidelines PDF documents to identify and extract relevant regulatory information. This prompt guides the automated parsing of complex regulatory documents.",
  kind: "prompt",
  supplement: [Prompt],
) <guidelines_retriever_data_extraction>

#let guidelines_retriever_prompts_system = read("code/guidelines_retriever_system_prompt.py")
#figure(
  code()[
    #raw(guidelines_retriever_prompts_system, lang: "python")
  ],
  caption: "System prompt for the Guidelines Retriever agent that defines its role in processing and interpreting Swiss federal and cantonal energy planning regulations. The prompt instructs the agent to analyze retrieved guideline documents, extract relevant regulatory requirements and contextualize them for specific municipal energy planning scenarios while ensuring compliance with applicable legislation.",
  kind: "prompt",
  supplement: [Prompt],
) <guidelines_retriever_system_prompt>
#pagebreak()
#let infer_pages_structureless_prompts_system = read("code/infer_no_structure_system.py")
#figure(
  code()[
    #raw(infer_pages_structureless_prompts_system, lang: "python")
  ],
  caption: "Data extraction prompt for the Municipal Citizen Profile component that processes unstructured citizen records to extract building-specific energy characteristics. This prompt guides the automated extraction of parcel data, energy consumption patterns, heating systems and renewable energy installations.",
  kind: "prompt",
  supplement: [Prompt],
) <municipal_citizen_profile>
#pagebreak()
#let generate_answer_factual_system_prompt = read("code/generate_answer_factual_system_prompt.py")
#figure(
  code()[
    #raw(generate_answer_factual_system_prompt, lang: "python")
  ],
  caption: "System prompt for the Strategy Planner agent when handling factual queries that request data analysis and information retrieval. This prompt defines the agent's role in processing geographical context, municipal data and guidelines to provide accurate, data-driven responses about current energy consumption, infrastructure status and resource availability without generating planning recommendations.",
  kind: "prompt",
  supplement: [Prompt],
) <generate_answer_factual_system_prompt>

#let generate_answer_factual_user_prompt = read("code/generate_answer_factual_user_prompt.py")
#figure(
  code()[
    #raw(generate_answer_factual_user_prompt, lang: "python")
  ],
  caption: "User prompt template for factual queries processed by the Strategy Planner agent, structuring how data requests are formatted with retrieved contextual information including geographical data, applicable guidelines and municipal characteristics. This template ensures consistent data presentation and accurate information synthesis for informational energy planning queries.",
  kind: "prompt",
  supplement: [Prompt],
) <generate_answer_factual_user_prompt>
#pagebreak()
#let generate_answer_actionable_system_prompt = read("code/generate_answer_actionable_system_prompt.py")
#figure(
  code()[
    #raw(generate_answer_actionable_system_prompt, lang: "python")
  ],
  caption: "System prompt for the Strategy Planner agent when handling actionable queries that require strategic planning recommendations. This prompt defines the agent's role in synthesizing geographical data, regulatory guidelines and municipal profiles to generate comprehensive energy planning strategies, implementation roadmaps and specific actionable recommendations tailored to the municipality's context and constraints.",
  kind: "prompt",
  supplement: [Prompt],
) <generate_answer_actionable_system_prompt>
#pagebreak()
#let generate_answer_actionable_user_prompt = read("code/generate_answer_actionable_user_prompt.py")
#figure(
  code()[
    #raw(generate_answer_actionable_user_prompt, lang: "python")
  ],
  caption: "User prompt template for actionable queries processed by the Strategy Planner agent, structuring how planning requests are formatted with comprehensive contextual information including geographical potential, regulatory constraints, municipal profiles and user preferences. This template enables the generation of tailored strategic recommendations for municipal energy planning implementation.",
  kind: "prompt",
  supplement: [Prompt],
) <generate_answer_actionable_user_prompt>

#let critic_answer_system_prompt = read("code/critic_answer_system_prompt.py")
#figure(
  code()[
    #raw(critic_answer_system_prompt, lang: "python")
  ],
  caption: "System prompt for the Critic agent that performs quality assurance on generated responses before delivery to users. This prompt defines the agent's role in evaluating responses for mathematical accuracy, logical consistency, proper data interpretation and alignment with guidelines, outputting a boolean decision on whether the response meets quality standards and should be delivered or regenerated.",
  kind: "prompt",
  supplement: [Prompt],
) <critic_answer_system_prompt>
#pagebreak()
#let benchmark_system_prompt = read("code/benchmark_system_prompt.py")
#figure(
  code()[
    #raw(benchmark_system_prompt, lang: "python")
  ],
  caption: "System prompt for the G-eval benchmarking framework that defines the evaluation criteria and scoring methodology for automated assessment of AI agent responses. This prompt instructs the evaluator model to assess responses across four key dimensions: data interpretation accuracy, methodology alignment with energy planning principles, municipal relevance and technical compliance, using a standardized 1-5 scoring scale.",
  kind: "prompt",
  supplement: [Prompt],
) <benchmark_system_prompt>

#pagebreak()

#heavy-title("B. Conversations", top: 1em, bottom: 1em)

#let example_conversation = read("code/conversation_example.md")
#figure(
  code()[
    #raw(example_conversation, lang: "md")
  ],
  caption: "Example conversation demonstrating the system's conversational capabilities. This report addresses the municipality of Ayent. It is only mentioned at the very beginning of the conversation and not needed further on.",
  kind: "conversation",
  supplement: [Conversation],
) <example_conversation>
#pagebreak()
#let bias_conversation = read("code/conversation_bias.md")
#figure(
  code()[
    #raw(bias_conversation, lang: "md")
  ],
  caption: "Example conversation demonstrating the system's vulnerability to generating biased or potentially misleading results when fabricating specific numerical figures in the absence of complete data.",
  kind: "conversation",
  supplement: [Conversation],
) <fake_data>
#pagebreak()
#let evaluation_conversation = read("code/dataset_evaluation.md")
#figure(
  code()[
    #raw(evaluation_conversation, lang: "md")
  ],
  caption: "Complete conversation transcript showing the responses of the system to all nine test dataset prompts for municipal energy planning in Sion. This comprehensive evaluation demonstrates the system's performance across different query types including energy consumption analysis, future projections, efficiency measures, renewable potential assessment and infrastructure mapping requests, providing the primary data for both expert assessment and automated benchmarking evaluation.",
  kind: "conversation",
  supplement: [Conversation],
) <evaluation_data>

#pagebreak()

#heavy-title("C. Equations", top: 1em, bottom: 1em)

#set math.equation(numbering: "1.")
#show math.equation.where(block: true): set align(center)

The following equation defines the sample mean (arithmetic mean) #ref(<ArithmeticMean2025>).
$
  overline(x) = frac(1, N)sum_(i=1)^N x_i
$<sample_mean>
Where $overline(x)$ is the sample mean, $N$ the number of samples and $x_i$ the $i_"th"$ sample tile.

#align(center, line(length: 90%, stroke: 0.5pt))

This equation provides the definition of the sample standard deviation #ref(<UnbiasedEstimationStandard2025>).
$
  s = sqrt(frac(sum_(i=1)^N (x_i-overline(x))², N-1))
$<std>
Where $s$ is the sample standard deviation of a single tile.

#align(center, line(length: 90%, stroke: 0.5pt))

The following equation defines the confidence interval for a single tile #ref(<NormalDistribution2025>).
$
  "Confidence interval for a single tile" = overline(x) plus.minus t_(alpha/2, N-1) dot frac(s, sqrt(N))
$ <confidence_interval>
Where $t_(alpha/2, N-1)$ is the critical value from the T-distribution for confidence level $1-alpha$ and $N-1$ degrees of freedom.

#align(center, line(length: 90%, stroke: 0.5pt))

This equation provides the definition of the cosine similarity #ref(<CosineSimilarity2025>).
$
  "Cosine similarity" := cos(theta) = frac("A" dot "B", norm("A")norm("B")) = frac(sum_(i=1)^n A_i B_i, sqrt(sum_(i=1)^n A_i²) dot sqrt(sum_(i=1)^n B_i²))
$ <cosine_sim>
Where $theta$ is the angle between A and B, two $n$-dimensional vectors and $A_i, B_i$ the $i_"th"$ components of vectors A and B.

#align(center, line(length: 90%, stroke: 0.5pt))

The following equation defines the quartile coefficient of dispersion #ref(<QuartileCoefficientDispersion2025>).
$
  "Quartile coefficient of dispersion" := frac(frac(1, 2)"IQR", frac(Q_3+Q_1, 2)) = frac(frac(1, 2)(Q_3-Q_1), frac(Q_3+Q_1, 2)) = frac(Q_3-Q_1, Q_3+Q_1)
$ <qcd>
Where $"IQR"$ is the interquartile range and $Q_1, Q_3$ the first and third quartiles, respectively.

#align(center, line(length: 90%, stroke: 0.5pt))

This equation provides the definition of Welford's online algorithm #ref(<AlgorithmsCalculatingVariance2025>).
$
  overline(x)_n = overline(x)_(n-1) + frac(x_n-overline(x)_(n-1), n) \
  M_(2,n) = M_(2,n-1)+(x_n-overline(x)_(n-1))(x_n-overline(x)_n) \
  s_n² =frac(M_(2,n), n-1)
$ <welford>
Where $overline(x)_n$ denotes the sample mean of the first $n$ samples, $M_(2,n)$ is the sum of squares of differences from the current mean and $s_n²$ the unbiased sample variance.
#pagebreak()
The following equation defines the z-score #ref(<StandardScore2025>).
$
  z = frac(x - mu, sigma)
$ <zscore>
Where $z$ is the standard score, $mu$ is the mean of the population and $sigma$ is the standard deviation of the population.

#align(center, line(length: 90%, stroke: 0.5pt))

This equation provides the heuristic used to approximate the runtime of a 10 W LED, in minutes.
$
  "Runtime"_"LED 10 W" = frac(h(x), 10 * 60) " "["minutes"]
$ <token_consumption>
Where $h(x) = 3x$ is the heuristic meant to approximate the energy cost of $x$ tokens in joules.

#align(center, line(length: 90%, stroke: 0.5pt))

The following equation defines the entropy, in information theory #ref(<EntropyInformationTheory2025>).
$
  "H"(X) := -sum_(x in cal(X))p(x)ln p(x) " "["nats"]
$ <entropy>
Where $"H"(X)$ is the entropy of the discrete random variable $X$ which takes value $x$ within the set $cal(X)$ and is distributed according to $p: cal(X) arrow [0,1]$.

#align(center, line(length: 90%, stroke: 0.5pt))

This equation demonstrates the calculation of the maximal entropy for the LLM-as-a-judge benchmarking criteria.

Let $X$ be a discrete random variable that takes value in the set $cal(X):={1,2,3,4,5}$ and suppose $X$ is uniformly distributed over $cal(X) ==> p(x) = frac(1, 5) forall x in cal(X)$, then, the entropy of $X$ is:
$
  "H"(X) = -sum_(x in cal(X))p(x)"ln"p(x)=-5 dot frac(1, 5)"ln"(frac(1, 5))="ln"(5) approx 1.61 "nats".
$ <entropy_upper_bound>

#align(center, line(length: 90%, stroke: 0.5pt))

The following equation defines Spearman's rank correlation coefficient #ref(<SpearmansRankCorrelation2025>).
$
  "Spearman coefficient of correlation" = frac("cov"["R"[X]", R"[Y]], sigma_("R"[X])sigma_("R"[Y]))
$ <spearman>
Where $"R"[X]$ and $"R"[Y]$ are the ranks of scores $(X_i,Y_i)$, $"cov"["R"[X]", R"[Y]]$ is the covariance of the rank variables and $sigma_("R"[X]),sigma_("R"[Y])$ are the standard deviations of the rank variables.

// Table of acronyms, NOT COMPULSORY
#print-index(
  title: heavy-title(i18n(doc_language, "acronym-table-title"), mult: 1, top: 1em, bottom: 1em),
  sorted: "up",
  delimiter: " : ",
  row-gutter: 0.7em,
  outlined: false,
)

// Table of listings
#table-of-figures()
