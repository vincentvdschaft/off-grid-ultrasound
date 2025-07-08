from pathlib import Path

import numpy as np


class LatexCell:
    def __init__(self, value, bold=False):
        self.value = value
        self.bold = bold

    def latex(self, format_fn=str):
        if self.bold:
            return r"\textbf{" + format_fn(self.value) + "}"
        return format_fn(self.value)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        assert isinstance(value, (str, int, float)) or value is None
        self._value = value

    def __float__(self):
        return float(self.value)

    def __gt__(self, other):
        return float(self.value) > float(other)

    def __lt__(self, other):
        return float(self.value) < float(other)

    def __eq__(self, other):
        return self.value == other.value


class LatexTable:
    def __init__(
        self,
        array,
        colnames=None,
        rownames=None,
        bold_rownames=False,
        bold_colnames=False,
        caption="",
        label="tab:latex_table",
        midrule_idx=None,
        colname_prefix="",
    ):
        self.cells = np.empty(shape=array.shape, dtype=object)
        self.n_rows = self.cells.shape[0]
        self.n_cols = self.cells.shape[1]
        for row in range(self.n_rows):
            for col in range(self.n_cols):
                self.cells[row, col] = LatexCell(array[row, col])

        self.colnames = colnames
        self.rownames = rownames
        self.bold_rownames = bold_rownames
        self.bold_colnames = bold_colnames
        self.caption = caption
        self.label = label
        self.midrule_idx = [] if midrule_idx is None else midrule_idx
        self.colname_prefix = colname_prefix

    def to_latex(self, format_fn=str):
        if self.bold_rownames:
            rownames = self._make_bold(self.rownames)
        else:
            rownames = self.rownames
        if self.bold_colnames:
            colnames = self._make_bold(self.colnames)
        else:
            colnames = self.colnames

        latex = ""
        latex += r"\begin{table}" + "\n"
        latex += r"\centering" + "\n"
        latex += r"\caption{" + self.caption + "}" + "\n"
        latex += r"\label{" + self.label + "}" + "\n"
        latex += r"\begin{tabular}{"
        latex += "l" + "c" * self.n_cols
        latex += "}\n"
        latex += r"\toprule" + "\n"
        latex += self.colname_prefix
        if colnames is not None:
            if rownames is not None:
                latex += " & "

            latex += " & ".join(colnames) + r"\\" + "\n"

        latex += r"\midrule" + "\n"
        for row in range(self.n_rows):
            if rownames is not None:
                latex += rownames[row] + " & "
            latex += (
                " & ".join([cell.latex(format_fn) for cell in self.cells[row]]) + r"\\"
            )
            if row in self.midrule_idx:
                latex += r"\midrule" + "\n"
            else:
                latex += "\n"

        latex += r"\bottomrule"
        latex += "\n"
        latex += r"\end{tabular}"
        latex += "\n"
        latex += r"\end{table}" + "\n"
        return latex

    @staticmethod
    def _make_bold(cells):
        if cells is None:
            return None
        if isinstance(cells, str):
            return f"\\textbf{{{cells}}}"
        return [f"\\textbf{{{cell}}}" for cell in cells]

    @property
    def n_cols_incnames(self):
        return self.n_cols + 1 if self.colnames is not None else self.n_cols

    def make_best_bold(self, cells, criterion="lowest", decimals=2):
        if criterion == "lowest":

            def is_better(new, old):
                return new < old

        else:

            def is_better(new, old):
                return new > old

        # Find the best values

        best_value = None
        best_idx = set()
        for n, cell in enumerate(cells):
            if isinstance(cell.value, str):
                continue

            cell_value = cell.value

            # Round to decimals
            cell_value = round(cell_value, decimals)

            if best_value == None or is_better(cell_value, best_value):
                best_value = cell_value
                best_idx = {n}
            elif cell_value == best_value:
                best_idx.add(n)

        # Make the best values bold
        for idx in best_idx:
            cells[idx].bold = True

    def make_best_in_row_bold(self, criterion="lowest", decimals=2):
        for row in self.cells:
            self.make_best_bold(row, criterion, decimals)

    def make_best_in_col_bold(self, criterion="lowest", decimals=2):
        for col in self.cells.T:
            self.make_best_bold(col, criterion, decimals)


def create_latex_command(directory, name, value):
    """Add the latex command to the file. If the command already exists, update the value."""
    file_path = Path(directory) / "auto_macros.tex"
    try:
        with file_path.open("r", newline="\n", encoding="ascii") as f:
            lines = f.readlines()
    except FileNotFoundError:
        lines = []

    # Replace digits with words (1 -> one, 2 -> two, etc.)
    new_name = _replace_digits(name)
    # Underscores are not allowed in latex commands
    new_name = _remove_underscores(new_name)
    new_name = _remove_points(new_name)
    new_name = _remove_commas(new_name)
    name = new_name

    command = r"\newcommand{" + "\\" + name + r"}{" + str(value) + r"}"

    # Update the command if it already exists
    for n, line in enumerate(lines):
        if line.startswith(f"\\newcommand{{\\{name}}}"):
            lines[n] = command
            break
    # Add the command if it does not exist (if we did not break)
    else:
        lines.append(command)

    # Remove empty lines
    lines = [line.strip() for line in lines if line.strip()]

    with file_path.open("w", newline="\n", encoding="ascii") as f:
        for line in lines:
            print(line, file=f)


def _replace_digits(text):
    """Replace digits with words (1 -> one, 2 -> two, etc.)"""
    text = text.replace("1", "one")
    text = text.replace("2", "two")
    text = text.replace("3", "three")
    text = text.replace("4", "four")
    text = text.replace("5", "five")
    text = text.replace("6", "six")
    text = text.replace("7", "seven")
    text = text.replace("8", "eight")
    text = text.replace("9", "nine")
    text = text.replace("0", "zero")
    return text


def _remove_underscores(text):
    """Removes un"""
    return text.replace("_", "")


def _remove_points(text):
    return text.replace(".", "d")


def _remove_commas(text):
    return text.replace(",", "")
