# LangChain コンテキストエンジニアリング について

## コンテキストエンジニアリングとは

Andrej Karpathy（元OpenAI/Tesla）の定義:

> コンテキストエンジニアリングとは「次のステップに必要な**正しい情報**でコンテキストウィンドウを満たす繊細な技術と科学」

エージェントが失敗する原因は主に2つで、LLM自体の能力不足か、正しいコンテキストが渡されていないかです。ほとんどの場合は後者が原因です。LangChain 1.0 のミドルウェアは、このコンテキスト制御を体系的に行うための仕組みです。

---

## 3つのコンテキスト制御

```
エージェントループ:
  ┌──────────────────────────────────┐
  │                                  │
  │  ┌─ Model Context (一時的) ──┐   │
  │  │ プロンプト / メッセージ    │   │
  │  │ ツール / モデル / 出力形式 │   │
  │  └──────────────────────────┘   │
  │           ↓                      │
  │     モデル呼び出し                │
  │           ↓                      │
  │  ┌─ Life-cycle Context ─────┐   │
  │  │ ガードレール / ロギング   │   │
  │  │ 要約 / バリデーション     │   │
  │  └──────────────────────────┘   │
  │           ↓                      │
  │  ┌─ Tool Context (永続的) ──┐   │
  │  │ State / Store の読み書き  │   │
  │  │ Runtime Context の参照    │   │
  │  └──────────────────────────┘   │
  │           ↓                      │
  └──── ループ先頭に戻る ────────────┘
```

| コンテキスト | 永続性 | 何を制御するか |
|---|---|---|
| **Model Context** | 一時的（Transient） | LLMの1回の呼び出しに何を渡すか |
| **Tool Context** | 永続的（Persistent） | ツールが何にアクセスし何を生成するか |
| **Life-cycle Context** | 永続的（Persistent） | モデルとツールの間に何をするか |

---

## 3つのデータソース

| データソース | 別名 | スコープ | アクセス方法 |
|---|---|---|---|
| **Runtime Context** | 静的設定 | 会話スコープ | `request.runtime.context` |
| **State** | 短期メモリ | 会話スコープ | `request.state`, `state["messages"]` |
| **Store** | 長期メモリ | 会話横断 | `request.runtime.store` |

### Runtime Context

`invoke` 時に渡す追加情報。user_id、APIキー、権限、環境設定など。

```python
@dataclass
class AppContext:
    user_role: str
    user_id: str

agent.invoke(
    {"messages": [...]},
    context=AppContext(user_role="admin", user_id="user_001"),
)
```

### State

会話内のメッセージ履歴やツール結果。checkpointer で保存される。

```python
# ミドルウェアから
msg_count = len(request.state["messages"])

# ツールから
runtime.state["messages"]
```

### Store

会話をまたいで永続するデータ。namespace + key で整理。

```python
# ツールから
runtime.store.get(("preferences",), user_id)
runtime.store.put(("preferences",), user_id, data)
```

---

## Part 1: Model Context

LLM の1回の呼び出しに対する**一時的**な変更。State は変更されない。

### システムプロンプト（`@dynamic_prompt`）

```python
@dynamic_prompt
def smart_prompt(request: ModelRequest) -> str:
    role = request.runtime.context.user_role  # Runtime Context
    msg_count = len(request.messages)          # State

    base = "あなたはアシスタントです。"
    if role == "admin":
        base += "\n管理者権限があります。"
    if msg_count > 10:
        base += "\n簡潔に回答してください。"
    return base
```

### メッセージ注入（`@wrap_model_call`）

```python
@wrap_model_call
def inject_context(request, handler):
    extra = {"role": "user", "content": "【追加情報】地域: 日本"}
    messages = [*request.messages, extra]
    return handler(request.override(messages=messages))
```

`wrap_model_call` は一時的な変更（Transient）。State のメッセージ履歴は変わらない。

### ツール選択

```python
@wrap_model_call
def filter_tools(request, handler):
    role = request.runtime.context.user_role
    if role == "viewer":
        tools = [t for t in request.tools if t.name.startswith("read_")]
    else:
        tools = request.tools
    return handler(request.override(tools=tools))
```

### モデル切替

```python
@wrap_model_call
def select_model(request, handler):
    if len(request.messages) > 10:
        model = init_chat_model("openai:gpt-4.1")
    else:
        model = init_chat_model("openai:gpt-4.1-nano")
    return handler(request.override(model=model))
```

### レスポンスフォーマット

```python
@wrap_model_call
def select_format(request, handler):
    if len(request.messages) < 3:
        return handler(request.override(response_format=SimpleAnswer))
    else:
        return handler(request.override(response_format=DetailedAnswer))
```

---

## Part 2: Tool Context

ツールが State / Store / Runtime Context の読み書きを行う。**永続的**な影響を持つ。

### 読み取り

```python
@tool
def get_data(query: str, runtime: ToolRuntime[UserContext]) -> str:
    user_id = runtime.context.user_id    # Runtime Context
    state = runtime.state                 # State
    prefs = runtime.store.get(...)        # Store
    return "結果"
```

`runtime` パラメータはモデルからは見えない（hidden）。LLM はツールを呼ぶとき `query` だけを指定する。

### 書き込み

```python
@tool
def save_preference(data: dict, runtime: ToolRuntime[UserContext]) -> str:
    runtime.store.put(("preferences",), runtime.context.user_id, data)
    return "保存しました"
```

Store への書き込みは永続的。別の thread からも読み取れる。

---

## Part 3: Life-cycle Context

モデル呼び出しとツール実行の**間**でStateに永続的な変更を加える。

### ガードレール（`@after_model`）

モデルの応答をチェックし、不適切な内容をブロック。

```python
@after_model
@hook_config(can_jump_to=["end"])
def content_guardrail(state, runtime):
    content = state["messages"][-1].content
    if "password" in content.lower():
        return {
            "messages": [AIMessage("セキュリティ上お答えできません。")],
            "jump_to": "end",
        }
    return None
```

### 自動要約（SummarizationMiddleware）

```python
SummarizationMiddleware(
    model="openai:gpt-4.1-mini",
    trigger=("tokens", 4000),
    keep=("messages", 20),
)
```

閾値を超えると古いメッセージを自動的に要約して圧縮。State が永続的に変更される。

### ロギング・監視

全ライフサイクルイベント（agent_start → before_model → after_model → tool_call → after_agent）を記録。

---

## Transient vs Persistent の区別

これがコンテキストエンジニアリングの最も重要な概念です:

| | Transient（一時的） | Persistent（永続的） |
|---|---|---|
| 手段 | `@wrap_model_call` + `request.override(...)` | `@before_model` / `@after_model` が dict を返す |
| 影響 | その1回のモデル呼び出しだけ | State に保存され、以降のターンに影響 |
| 例 | メッセージ注入、ツールフィルタ、モデル切替 | 要約、メッセージ削除、ガードレール |

```
Transient: request.override(messages=...) → モデルに渡すだけ、State は変わらない
Persistent: return {"messages": [...]} → State が更新される
```

---

## 本番構成のパターン

```python
agent = create_agent(
    model="openai:gpt-4.1",
    tools=[...],
    middleware=[
        # 1. Life-cycle: 全体監視（最初に配置）
        ObservabilityMiddleware(),
        # 2. Model Context: 動的プロンプト
        smart_prompt,
        # 3. Model Context: ツールフィルタ
        filter_tools_by_role,
        # 4. Model Context: モデル切替
        select_model,
        # 5. Life-cycle: 自動要約
        SummarizationMiddleware(model="gpt-4.1-mini", trigger=("tokens", 4000)),
        # 6. Life-cycle: ガードレール
        content_guardrail,
    ],
    context_schema=AppContext,
    store=InMemoryStore(),         # 長期メモリ
    checkpointer=InMemorySaver(),  # 短期メモリ
)
```

---

## 参考リンク

- [Context engineering in agents 公式ドキュメント](https://docs.langchain.com/oss/python/langchain/context-engineering)
- [Context engineering 概念ガイド](https://docs.langchain.com/oss/python/concepts/context)
- [LangChain ブログ: Context Engineering for Agents](https://blog.langchain.com/context-engineering-for-agents/)
- [LangChain ブログ: Context Management for Deep Agents](https://blog.langchain.com/context-management-for-deepagents/)
- [langchain-ai/context_engineering (GitHub)](https://github.com/langchain-ai/context_engineering)
- [langchain-ai/how_to_fix_your_context (GitHub)](https://github.com/langchain-ai/how_to_fix_your_context)
