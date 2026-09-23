*Bygger ett litet sidoprojekt: Promptimize*

Hej allihopa! Har byggt ett litet fristående verktyg på fritiden som jag
tänkte dela, ifall någon annan tycker det är kul eller nyttigt.

*Vad det är*
Promptimize analyserar din egen prompthistorik från GitHub Copilot CLI
och/eller Claude Code (llm agnostisk) för en valfri period ("den här veckan", "senaste 30
dagarna" osv.) och genererar en interaktiv HTML-rapport med konkret,
sifferbaserad återkoppling på hur man promptar en AI-agent - inte generiska
tips, utan mönster i just dina egna konversationer.

*Varför*
 och tänkte: finns det något
sätt att faktiskt mäta om jag blir bättre på det över tid, och få konkreta,
underbyggda förbättringsförslag istället för känsla/magkänsla? Så blev det
detta.

*Vad rapporten innehåller*
- Fyra poängaxlar (0-100): Specificitet, Kontextförankring, Struktur/
  acceptanskriterier, Effektivitet (rättningar, förtydliganden, upprepning).
- Evidensbaserade rekommendationer - varje förslag citerar en egen siffra
  från din historik (t.ex. "rättningsfrekvensen är 3x högre på prompts utan
  filreferens, n=42") plus en förklaring av varför det spelar roll för hur
  LLM:er hanterar kontext.
- Progress-spårning över tid - varje körning sparas lokalt, så nästa
  rapport visar om du faktiskt förbättrat dig och om tidigare mål uppnåtts.
- "Avancerad analys" med lite roligare matematik: Markov-kedjor för
  konversationstillstånd, diskret varvtal (winding number) för cirkulära/
  olösta konversationer, "burstiness" i tidsmönster, en komprimerings-
  baserad komplexitetsproxy, och Heaps lag för ordförrådstillväxt. Alla
  formler och härledningar finns dokumenterade i README:n (appendix).
- Anomali-detektering (z-score mot din egen historik) och Mahalanobis-
  avstånd till din egen "optimala punkt" - inte en generisk baseline, utan
  kalibrerat mot din egen variation.
- Nytt: en "Leveling"-flik i stil med klassiska MMO-färdighetssystem - 7
  olika "skills" (specificitet, kontextförankring, struktur, effektivitet,
  tydlighet, kontextbevarande, ordförråd) som får XP och levlar 1-99 baserat
  på hur du promptar, plus en samlad "Total level" och en viktad "Prompt
  Level". XP är kumulativ för alltid och dubbelräknas aldrig även om man
  kör överlappande perioder.
- Opt-in jämförelse med kollegor ("Hiscores") - om man vill dela sin
  exporterade JSON (ingen rå prompttext, bara siffror) kan man ranka sig
  mot varandra, helt frivilligt och lokalt sammanslaget - inget skickas
  till någon server.
- Team-läge: flera personers exporter kan slås ihop till en anonymiserad
  gruppsammanställning (poängfördelning, svagaste axlar, vanliga
  anomalier) - också helt opt-in.

*Filosofi*
- 100% lokalt. Inget lämnar din maskin förutom Plotly-grafbiblioteket från
  CDN (bara för att rita graferna). Ingen server, inget konto, ingen
  telemetri.
- Noll tredjepartsberoenden för själva verktyget (bara Python-standard-
  bibliotek). Testsviten använder pytest i en lokal venv, men verktyget
  självt kräver ingenting extra.
- Varje siffra är spårbar till en formel - inget är en svart låda. README:n
  har fullständiga matematiska härledningar för alla mätvärden.

*Status*
Fortfarande work in progress och ett fritidsprojekt, men fullt
körbart redan nu med testsvit (86 tester och växande).

Repo: github.com/patrikdennis/promptimize

Säg gärna till om ni vill testa det eller har idéer på fler mätvärden -
väldigt öppet för input!
