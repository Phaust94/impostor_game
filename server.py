import os.path
import random
import urllib.parse
import enum

import numpy as np
import pydantic
import uvicorn
import fastapi
from jinja2 import Environment, FileSystemLoader
import pandas as pd


app = fastapi.FastAPI()

DATA_DIR = "data"

IMPOSTOR_RANGE_TABLE = [
    {"players": 2, "probabilities": {0: 1 / 2, 1: 1 / 2}},
    {"players": 3, "probabilities": {0: 1 / 3, 1: 2 / 3}},
    {"players": 4, "probabilities": {0: 2 / 8, 1: 5 / 8, 2: 1 / 8}},
    {"players": 5, "probabilities": {0: 1 / 10, 1: 6 / 10, 2: 3 / 10}},
]


class GameKind(enum.Enum):
    Noun = "noun"
    Question = "question"

    def fn(self) -> str:
        res = {
            self.Noun: "noun_game.csv",
            self.Question: "questions_game.csv",
        }[self]
        return res

    def add_real_q(self) -> bool:
        res = {
            self.Noun: False,
            self.Question: True,
        }[self]
        return res


class Question(pydantic.BaseModel):
    question_id: int
    text_en_real: str
    text_uk_real: str
    text_en_impostor: str
    text_uk_impostor: str


def load_questions(game: GameKind) -> list[Question]:
    q_df = pd.read_csv(os.path.join(DATA_DIR, game.fn()))
    q_df.columns = [
        "text_en_real",
        "text_uk_real",
        "text_en_impostor",
        "text_uk_impostor",
    ]
    q_df["question_id"] = range(len(q_df))
    questions = [Question(**row.to_dict()) for _, row in q_df.iterrows()]
    return questions


QUESTIONS_DICT = {kind: load_questions(kind) for kind in GameKind}


def get_question(question_id: int, kind: GameKind) -> Question:
    db = QUESTIONS_DICT[kind]
    question = db[question_id % (len(db))]
    return question


def get_question_assignments(
    question: Question, seed: int, players: int, question_order: int | None = None
) -> list[dict[str, str | int]]:
    info = [x for x in IMPOSTOR_RANGE_TABLE if x["players"] == players]
    if not info:
        raise ValueError(f"Invalid number of players: {players}")
    info = info[0]
    probs = info["probabilities"]
    random.seed(seed + (question_order or 0))
    n_impostors = random.choices(
        population=list(probs.keys()), k=1, weights=list(probs.values())
    )[0]
    impostors = set(random.choices(population=list(range(players)), k=n_impostors))
    questions = [
        {
            "player_id": i + 1,
            "question_en": (
                question.text_en_impostor if i in impostors else question.text_en_real
            ),
            "question_uk": (
                question.text_uk_impostor if i in impostors else question.text_uk_real
            ),
        }
        for i in range(players)
    ]
    return questions


def get_next_question_url(
    seed: int,
    players: int,
    kind: GameKind,
    next_question_order: int = None,
    player_id: int | None = None,
):
    if next_question_order is None:
        return None
    else:
        data = {
            "seed": seed,
            "players": players,
            "question_order": next_question_order,
            "kind": kind.value,
        }
        if player_id:
            data["player_id"] = player_id
        return f"/impostor/v2/random?{urllib.parse.urlencode(data)}"


def get_body(
    question: Question,
    question_assignments: list[dict[str, str | int]],
    seed: int,
    players: int,
    kind: GameKind,
    question_order: int = None,
    player_id: int | None = None,
) -> str:
    env = Environment(loader=FileSystemLoader(DATA_DIR))
    template = env.get_template("question_page.html")
    next_question_url = get_next_question_url(
        seed, players, kind, question_order + 1 if question_order else None, player_id
    )
    rendered_html = template.render(
        {
            "questions": question_assignments,
            "next_question_url": next_question_url,
            "real_question": {
                "question_uk": question.text_uk_real,
                "question_en": question.text_en_real,
            },
            "qn": question_order if question_order else 0,
            "seed": seed,
            "add_real_q": kind.add_real_q(),
        }
    )
    return rendered_html


@app.get("/impostor/v1/{question_id}", response_class=fastapi.responses.HTMLResponse)
async def get_question_by_number(
    question_id: int,
    seed: int,
    players: int,
    player_id: int | None = None,
    question_order: int = None,
    kind: GameKind = GameKind.Question,
) -> str:
    question = get_question(question_id, kind)
    try:
        question_assignments = get_question_assignments(
            question, seed, players, question_order
        )
    except ValueError as e:
        raise fastapi.HTTPException(status_code=400, detail=str(e))
    if player_id:
        question_assignments = [
            ass for ass in question_assignments if ass["player_id"] == player_id
        ]
    rendered_html = get_body(
        question, question_assignments, seed, players, kind, question_order, player_id
    )

    return rendered_html


def get_random_permutation(seed: int, kind: GameKind):
    arr = np.arange(len(QUESTIONS_DICT[kind]))
    np.random.seed(seed + 1)
    np.random.shuffle(arr)
    return arr.tolist()


@app.get("/impostor/v2/random", response_class=fastapi.responses.HTMLResponse)
async def get_question_random(
    seed: int,
    players: int,
    question_order: int,
    player_id: int | None = None,
    kind: GameKind = GameKind.Question,
) -> str:
    permutation = get_random_permutation(seed, kind)
    question_id = permutation[question_order % len(QUESTIONS_DICT[kind])]
    res = await get_question_by_number(
        question_id, seed, players, player_id, question_order=question_order, kind=kind
    )
    return res


def get_start_page_body(
    url_assignments: list[dict[str, str | int]],
) -> str:
    env = Environment(loader=FileSystemLoader(DATA_DIR))
    template = env.get_template("start_page.html")
    rendered_html = template.render({"player_info": url_assignments})
    return rendered_html


def get_start_page_assignments(
    players: int, seed: int, kind: GameKind
) -> list[dict[str, int | str]]:
    assignments = [
        {
            "player_id": i + 1,
            "url": "/impostor/v2/random?{}".format(
                urllib.parse.urlencode(
                    {
                        "seed": seed,
                        "players": players,
                        "question_order": 1,
                        "player_id": i + 1,
                        "kind": kind.value,
                    }
                )
            ),
        }
        for i in range(players)
    ]
    return assignments


@app.get("/impostor/v2/start", response_class=fastapi.responses.HTMLResponse)
async def start_page(
    players: int,
    seed: int | None = None,
    kind: GameKind = GameKind.Question,
) -> str:
    if seed is None:
        seed = random.randint(1, 1_000_000)
    assignments = get_start_page_assignments(players, seed, kind)
    res = get_start_page_body(assignments)
    return res


if __name__ == "__main__":
    dc = {kind: len(QUESTIONS_DICT[kind]) for kind in GameKind}
    print(f"Running game with {dc} questions")
    uvicorn.run(app, host="0.0.0.0", port=8000)
