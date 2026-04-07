from __future__ import annotations

import argparse
import io
import re
import zipfile
from pathlib import Path
from typing import Dict, List
from urllib.request import urlopen

import pandas as pd


UCI_DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00331/"
    "sentiment%20labelled%20sentences.zip"
)
EXPECTED_ROWS_DEFAULT = 3000
SOURCE_FILES = {
    "imdb": "sentiment labelled sentences/imdb_labelled.txt",
    "amazon": "sentiment labelled sentences/amazon_cells_labelled.txt",
    "yelp": "sentiment labelled sentences/yelp_labelled.txt",
}


def download_source_files(cache_dir: Path) -> Dict[str, Path]:
    """Baixa e extrai os arquivos do dataset UCI em cache local."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "sentiment_labelled_sentences.zip"

    if not zip_path.exists():
        print(f"[INFO] Baixando dataset UCI de: {UCI_DATASET_URL}")
        with urlopen(UCI_DATASET_URL) as response:
            zip_path.write_bytes(response.read())

    extracted_paths: Dict[str, Path] = {}
    with zipfile.ZipFile(zip_path, "r") as zf:
        for source_name, archive_path in SOURCE_FILES.items():
            output_path = cache_dir / Path(archive_path).name
            if not output_path.exists():
                output_path.write_bytes(zf.read(archive_path))
            extracted_paths[source_name] = output_path

    return extracted_paths


def load_single_source(file_path: Path, source_name: str) -> pd.DataFrame:
    """Lê um arquivo origem (texto + rótulo) em DataFrame padronizado."""
    rows: List[Dict[str, str]] = []
    skipped_lines = 0

    with file_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n").rstrip("\r")
            if not line.strip():
                continue
            if "\t" not in line:
                skipped_lines += 1
                continue

            comment, label = line.rsplit("\t", 1)
            rows.append({"comment": comment, "class": label, "source": source_name})

    if skipped_lines:
        print(f"[WARN] {source_name}: {skipped_lines} linha(s) ignorada(s) por formato invalido.")

    return pd.DataFrame(rows, columns=["comment", "class", "source"])


def clean_comment(text: str) -> str:
    """Normaliza caracteres de controle e espaçamento para evitar erro de parse."""
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\x00", " ")
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def consolidate_and_clean(sources: Dict[str, Path]) -> pd.DataFrame:
    """Consolida IMDb/Amazon/Yelp e aplica limpeza e validações de classe."""
    frames: List[pd.DataFrame] = []
    for source_name, file_path in sources.items():
        frames.append(load_single_source(file_path, source_name))

    df = pd.concat(frames, ignore_index=True)

    # Limpeza estrutural
    df = df.dropna(subset=["comment", "class"])
    df["comment"] = df["comment"].map(clean_comment)
    df = df[df["comment"].str.len() > 0]

    # Garante rótulos válidos e explícitos como 0/1
    df["class"] = pd.to_numeric(df["class"], errors="coerce")
    df = df[df["class"].isin([0, 1])]
    df["class"] = df["class"].astype(int)

    df = df.reset_index(drop=True)
    return df[["comment", "class", "source"]]


def escape_arff_string(value: str) -> str:
    """Escapa string conforme esperado pelo Weka para atributo string."""
    value = clean_comment(value)
    value = value.replace("\\", "\\\\")
    value = value.replace('"', '\\"')
    return f'"{value}"'


def export_to_arff(df: pd.DataFrame, output_file: Path) -> None:
    """Exporta DataFrame para ARFF com schema requerido pelo trabalho."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8", newline="\n") as f:
        f.write("@relation sentiment_analysis\n\n")
        f.write("@attribute comment string\n")
        f.write("@attribute class {0,1}\n\n")
        f.write("@data\n")
        for row in df.itertuples(index=False, name=None):
            f.write(f"{escape_arff_string(row[0])},{row[1]}\n")


def export_to_csv(df: pd.DataFrame, output_file: Path) -> None:
    """Gera CSV auxiliar para inspeção e auditoria da consolidação."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False, encoding="utf-8")


def print_dataset_report(df: pd.DataFrame, expected_rows: int) -> None:
    """Mostra estatísticas rápidas para validação do pipeline."""
    print("\n[INFO] Estatísticas da base consolidada")
    print(f"- Total de linhas: {len(df)}")
    print("- Distribuição de classes:")
    print(df["class"].value_counts().sort_index().to_string())
    print("- Fontes:")
    print(df["source"].value_counts().to_string())

    if len(df) != expected_rows:
        print(
            f"[WARN] Total diferente de {expected_rows}. "
            "Verifique conectividade/arquivo fonte e regras de limpeza."
        )


def run_pipeline(output_dir: Path, cache_dir: Path, expected_rows: int) -> None:
    sources = download_source_files(cache_dir=cache_dir)
    df = consolidate_and_clean(sources)

    arff_path = output_dir / "sentiment_labelled_sentences.arff"
    csv_path = output_dir / "sentiment_labelled_sentences_clean.csv"

    export_to_arff(df, arff_path)
    export_to_csv(df, csv_path)
    print_dataset_report(df, expected_rows=expected_rows)

    print("\n[OK] Arquivos gerados:")
    print(f"- ARFF: {arff_path}")
    print(f"- CSV : {csv_path}")


def preview_first_lines(file_path: Path, n: int = 10) -> None:
    print(f"\n[INFO] Primeiras {n} linhas de {file_path}:")
    with file_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            print(line.rstrip("\n"))


def self_test() -> None:
    """Teste rápido offline para validar a serialização ARFF."""
    sample = pd.DataFrame(
        {
            "comment": [
                "I loved this movie!",
                "Bad\nexperience, would not buy again.",
                'The product said "premium" but failed.',
            ],
            "class": [1, 0, 0],
            "source": ["imdb", "amazon", "yelp"],
        }
    )

    out = io.StringIO()
    out.write("@relation sentiment_analysis\n\n")
    out.write("@attribute comment string\n")
    out.write("@attribute class {0,1}\n\n")
    out.write("@data\n")
    for row in sample.itertuples(index=False, name=None):
        out.write(f"{escape_arff_string(row[0])},{row[1]}\n")

    content = out.getvalue()
    assert "@relation sentiment_analysis" in content
    assert '"Bad experience, would not buy again.",0' in content
    assert '\\"premium\\"' in content
    print("[OK] Self-test concluído com sucesso.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pipeline para consolidar Sentiment Labelled Sentences e exportar ARFF para Weka."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data") / "processed",
        help="Diretório de saída para ARFF/CSV.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data") / "raw",
        help="Diretório de cache para ZIP e arquivos extraídos do UCI.",
    )
    parser.add_argument(
        "--expected-rows",
        type=int,
        default=EXPECTED_ROWS_DEFAULT,
        help=(
            "Total esperado após consolidação (padrão UCI oficial: 3000). "
            "Se seu enunciado exigir 3003, informe --expected-rows 3003."
        ),
    )
    parser.add_argument(
        "--preview-lines",
        type=int,
        default=10,
        help="Quantidade de linhas exibidas no preview do ARFF após geração.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Executa apenas um teste rápido local, sem download.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return 0

    run_pipeline(
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        expected_rows=args.expected_rows,
    )
    preview_first_lines(
        file_path=args.output_dir / "sentiment_labelled_sentences.arff",
        n=args.preview_lines,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

