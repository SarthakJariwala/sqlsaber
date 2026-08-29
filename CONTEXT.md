# SQLsaber

SQLsaber turns conversational data work into SQL results and durable analytical outputs while keeping large or binary data outside model history.

## Language

**Query Result**:
The complete immutable row set returned by one successful SQL execution. It is an analysis input and is distinct from an artifact.
_Avoid_: Artifact, output file

**Artifact**:
One typed immutable file produced by a capability operation for people or applications to retain and use.
_Avoid_: Query result, attachment

**Artifact Publication**:
A related collection of artifacts produced by one capability operation.
_Avoid_: Bundle, upload

**Artifact Store**:
The owner of immutable artifact publication and authorized artifact retrieval.
_Avoid_: Artifact publisher, blob store

**Artifact Reference**:
Durable metadata that identifies an artifact without containing its bytes.
_Avoid_: Artifact, signed URL

**SDK Conversation**:
The `SQLSaber` instance that owns one conversation's completed history, agent
lifecycle, optional thread persistence, and managed resources. It is the canonical
conversation lifecycle used by the CLI and TUI.
_Avoid_: Embedded wrapper, stateless query helper

**Client**:
An application such as the CLI, TUI, script, notebook, or web backend that owns
user input and presentation. Clients must use `SQLSaber` for core agent behavior,
conversation history, and thread lifecycle.
_Avoid_: Alternate agent, conversation owner
