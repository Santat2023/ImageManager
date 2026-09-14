import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import ImageTk

from image_manager_core import (
    DEFAULT_RESOLUTION,
    create_collection,
    delete_collection,
    list_collections,
    load_image_from_s3,
    process_directory,
    search_images,
)


class ImageManagerApp(tk.Tk):

    def __init__(self):

        super().__init__()

        self.title(
            "Image Manager — Qdrant RAG v4"
        )

        self.geometry(
            "1000x800"
        )

        self.minsize(
            800,
            600
        )

        self.notebook = ttk.Notebook(
            self
        )

        self.notebook.pack(
            fill="both",
            expand=True
        )

        self.tab_collections = ttk.Frame(
            self.notebook
        )

        self.tab_upload = ttk.Frame(
            self.notebook
        )

        self.tab_search = ttk.Frame(
            self.notebook
        )

        self.notebook.add(
            self.tab_collections,
            text="🗑 Коллекции"
        )

        self.notebook.add(
            self.tab_upload,
            text="⬆ Загрузка"
        )

        self.notebook.add(
            self.tab_search,
            text="🔎 Поиск"
        )

        self.init_collections_tab()
        self.init_upload_tab()
        self.init_search_tab()

    # ========================================================
    # COLLECTIONS
    # ========================================================

    def init_collections_tab(self):

        frame = self.tab_collections

        ttk.Label(
            frame,
            text="Существующие коллекции:"
        ).pack(
            anchor="w",
            padx=10,
            pady=5
        )

        self.collections_var = (
            tk.StringVar(
                value=list_collections()
            )
        )

        self.collections_listbox = (
            tk.Listbox(
                frame,
                listvariable=
                    self.collections_var,
                height=10
            )
        )

        self.collections_listbox.pack(
            fill="x",
            padx=10
        )

        ttk.Label(
            frame,
            text="Название новой коллекции:"
        ).pack(
            anchor="w",
            padx=10,
            pady=5
        )

        self.new_collection_entry = (
            ttk.Entry(frame)
        )

        self.new_collection_entry.pack(
            fill="x",
            padx=10
        )

        ttk.Button(
            frame,
            text="Создать коллекцию",
            command=
                self.create_collection_ui
        ).pack(
            pady=5,
            padx=10,
            anchor="w"
        )

        ttk.Button(
            frame,
            text="Удалить выбранную",
            command=
                self.delete_collection_ui
        ).pack(
            pady=5,
            padx=10,
            anchor="w"
        )

        ttk.Button(
            frame,
            text="Обновить список",
            command=
                self.refresh_collections
        ).pack(
            pady=5,
            padx=10,
            anchor="w"
        )

        info = (
            "Структура коллекции v4:\n\n"
            "auto = автоматическое описание BLIP → CLIP\n"
            "manual = ручная аннотация TXT → CLIP\n\n"
            "Если рядом с изображением нет TXT,\n"
            "для него используется только auto-вектор."
        )

        ttk.Label(
            frame,
            text=info,
            justify="left"
        ).pack(
            anchor="w",
            padx=10,
            pady=20
        )

    def refresh_collections(self):

        collections = list_collections()

        self.collections_var.set(
            collections
        )

        if hasattr(
            self,
            "upload_collection_combo"
        ):

            self.upload_collection_combo[
                "values"
            ] = collections

        if hasattr(
            self,
            "search_collection_combo"
        ):

            self.search_collection_combo[
                "values"
            ] = collections

    def create_collection_ui(self):

        name = (
            self.new_collection_entry
            .get()
            .strip()
        )

        if not name:

            messagebox.showerror(
                "Ошибка",
                "Введите название коллекции."
            )

            return

        try:

            result = create_collection(
                name
            )

            messagebox.showinfo(
                "Коллекция",
                result
            )

            self.new_collection_entry.delete(
                0,
                tk.END
            )

            self.refresh_collections()

        except Exception as e:

            messagebox.showerror(
                "Ошибка",
                str(e)
            )

    def delete_collection_ui(self):

        selection = (
            self.collections_listbox
            .curselection()
        )

        if not selection:

            messagebox.showerror(
                "Ошибка",
                "Выберите коллекцию."
            )

            return

        name = (
            self.collections_listbox
            .get(selection[0])
        )

        answer = messagebox.askyesno(
            "Удаление коллекции",
            f"Удалить коллекцию «{name}»?"
        )

        if not answer:
            return

        try:

            result = delete_collection(
                name
            )

            messagebox.showinfo(
                "Коллекция",
                result
            )

            self.refresh_collections()

        except Exception as e:

            messagebox.showerror(
                "Ошибка",
                str(e)
            )

    # ========================================================
    # UPLOAD
    # ========================================================

    def init_upload_tab(self):

        frame = self.tab_upload

        ttk.Label(
            frame,
            text="Каталог с изображениями:"
        ).pack(
            anchor="w",
            padx=10,
            pady=5
        )

        directory_frame = ttk.Frame(
            frame
        )

        directory_frame.pack(
            fill="x",
            padx=10
        )

        self.dir_entry = ttk.Entry(
            directory_frame
        )

        self.dir_entry.pack(
            side="left",
            fill="x",
            expand=True
        )

        ttk.Button(
            directory_frame,
            text="Выбрать папку",
            command=
                self.choose_directory
        ).pack(
            side="left",
            padx=(5, 0)
        )

        ttk.Label(
            frame,
            text=
                "Минимальная сторона изображения (px):"
        ).pack(
            anchor="w",
            padx=10,
            pady=5
        )

        self.resolution_spin = (
            ttk.Spinbox(
                frame,
                from_=64,
                to=2048,
                increment=1
            )
        )

        self.resolution_spin.set(
            DEFAULT_RESOLUTION
        )

        self.resolution_spin.pack(
            fill="x",
            padx=10
        )

        ttk.Label(
            frame,
            text="Коллекция:"
        ).pack(
            anchor="w",
            padx=10,
            pady=5
        )

        self.upload_collection_var = (
            tk.StringVar()
        )

        self.upload_collection_combo = (
            ttk.Combobox(
                frame,
                textvariable=
                    self.upload_collection_var,
                values=list_collections(),
                state="readonly"
            )
        )

        self.upload_collection_combo.pack(
            fill="x",
            padx=10
        )

        ttk.Label(
            frame,
            text=(
                "Ручная разметка:\n"
                "image.jpg → image.txt\n\n"
                "Если TXT существует, его текст "
                "используется для создания "
                "manual-вектора."
            ),
            justify="left"
        ).pack(
            anchor="w",
            padx=10,
            pady=15
        )

        self.progressbar = (
            ttk.Progressbar(
                frame,
                mode="determinate"
            )
        )

        self.progressbar.pack(
            fill="x",
            pady=10,
            padx=10
        )

        ttk.Button(
            frame,
            text=
                "Загрузить и проиндексировать изображения",
            command=self.upload_images
        ).pack(
            pady=10,
            padx=10,
            anchor="w"
        )

    def choose_directory(self):

        directory = (
            filedialog.askdirectory(
                title=
                    "Выберите папку с изображениями"
            )
        )

        if directory:

            self.dir_entry.delete(
                0,
                tk.END
            )

            self.dir_entry.insert(
                0,
                directory
            )

    def progress(
        self,
        current,
        total
    ):

        self.progressbar[
            "maximum"
        ] = total

        self.progressbar[
            "value"
        ] = current

        self.update_idletasks()

    def upload_images(self):

        directory = (
            self.dir_entry
            .get()
            .strip()
        )

        collection = (
            self.upload_collection_var
            .get()
            .strip()
        )

        try:

            resolution = int(
                self.resolution_spin.get()
            )

        except ValueError:

            messagebox.showerror(
                "Ошибка",
                "Некорректное значение разрешения."
            )

            return

        if not collection:

            messagebox.showerror(
                "Ошибка",
                "Выберите коллекцию."
            )

            return

        if (
            not directory
            or not os.path.exists(directory)
        ):

            messagebox.showerror(
                "Ошибка",
                "Указанный каталог не существует."
            )

            return

        try:

            process_directory(
                directory,
                resolution,
                collection,
                self.progress
            )

            messagebox.showinfo(
                "Готово",
                "Изображения успешно проиндексированы."
            )

        except Exception as e:

            messagebox.showerror(
                "Ошибка загрузки",
                str(e)
            )

    # ========================================================
    # SEARCH
    # ========================================================

    def init_search_tab(self):

        outer = ttk.Frame(
            self.tab_search
        )

        outer.pack(
            fill="both",
            expand=True
        )

        # Canvas
        self.search_canvas = tk.Canvas(
            outer,
            highlightthickness=0
        )

        # Scrollbar
        scrollbar = ttk.Scrollbar(
            outer,
            orient="vertical",
            command=
                self.search_canvas.yview
        )

        self.search_canvas.configure(
            yscrollcommand=
                scrollbar.set
        )

        scrollbar.pack(
            side="right",
            fill="y"
        )

        self.search_canvas.pack(
            side="left",
            fill="both",
            expand=True
        )

        # Внутренний Frame
        self.search_content = ttk.Frame(
            self.search_canvas
        )

        self.search_window = (
            self.search_canvas.create_window(
                (0, 0),
                window=self.search_content,
                anchor="nw"
            )
        )

        self.search_content.bind(
            "<Configure>",
            self._update_scroll_region
        )

        self.search_canvas.bind(
            "<Configure>",
            self._resize_search_content
        )

        # Windows mouse wheel
        self.search_canvas.bind_all(
            "<MouseWheel>",
            self.on_mousewheel,
            add="+"
        )

        frame = self.search_content

        # Collection

        ttk.Label(
            frame,
            text="Коллекция:"
        ).pack(
            anchor="w",
            padx=10,
            pady=5
        )

        self.search_collection_var = (
            tk.StringVar()
        )

        self.search_collection_combo = (
            ttk.Combobox(
                frame,
                textvariable=
                    self.search_collection_var,
                values=list_collections(),
                state="readonly"
            )
        )

        self.search_collection_combo.pack(
            fill="x",
            padx=10
        )

        # Query

        ttk.Label(
            frame,
            text="Поисковый запрос:"
        ).pack(
            anchor="w",
            padx=10,
            pady=5
        )

        self.query_entry = ttk.Entry(
            frame
        )

        self.query_entry.pack(
            fill="x",
            padx=10
        )

        # Weights

        weights_frame = ttk.LabelFrame(
            frame,
            text=
                "Весовые коэффициенты поиска"
        )

        weights_frame.pack(
            fill="x",
            padx=10,
            pady=10
        )

        ttk.Label(
            weights_frame,
            text=
                "Вес ручной разметки:"
        ).grid(
            row=0,
            column=0,
            sticky="w",
            padx=10,
            pady=5
        )

        self.manual_weight_spin = (
            ttk.Spinbox(
                weights_frame,
                from_=0.0,
                to=1.0,
                increment=0.1,
                width=10
            )
        )

        self.manual_weight_spin.set(
            "0.7"
        )

        self.manual_weight_spin.grid(
            row=0,
            column=1,
            padx=10,
            pady=5
        )

        ttk.Label(
            weights_frame,
            text=
                "Вес автоматического описания:"
        ).grid(
            row=1,
            column=0,
            sticky="w",
            padx=10,
            pady=5
        )

        self.auto_weight_spin = (
            ttk.Spinbox(
                weights_frame,
                from_=0.0,
                to=1.0,
                increment=0.1,
                width=10
            )
        )

        self.auto_weight_spin.set(
            "0.3"
        )

        self.auto_weight_spin.grid(
            row=1,
            column=1,
            padx=10,
            pady=5
        )

        ttk.Label(
            weights_frame,
            text=(
                "Рекомендуемая настройка:\n"
                "ручная разметка = 0.7\n"
                "автоматическое описание = 0.3\n\n"
                "Весовые коэффициенты "
                "нормализуются автоматически."
            ),
            justify="left"
        ).grid(
            row=0,
            column=2,
            rowspan=2,
            padx=20,
            pady=5
        )

        # Top K

        ttk.Label(
            frame,
            text=
                "Количество результатов:"
        ).pack(
            anchor="w",
            padx=10,
            pady=5
        )

        self.top_k_spin = (
            ttk.Spinbox(
                frame,
                from_=1,
                to=50,
                increment=1,
                width=10
            )
        )

        self.top_k_spin.set(
            "5"
        )

        self.top_k_spin.pack(
            anchor="w",
            padx=10
        )

        ttk.Button(
            frame,
            text="🔎 Выполнить поиск",
            command=
                self.search_images_ui
        ).pack(
            pady=10,
            padx=10,
            anchor="w"
        )

        ttk.Label(
            frame,
            text="Результаты:"
        ).pack(
            anchor="w",
            padx=10,
            pady=(10, 5)
        )

        self.results_frame = ttk.Frame(
            frame
        )

        self.results_frame.pack(
            fill="x",
            padx=10,
            pady=10
        )

    # ========================================================
    # SCROLLING
    # ========================================================

    def _update_scroll_region(
        self,
        _event=None
    ):

        self.search_canvas.configure(
            scrollregion=
                self.search_canvas.bbox(
                    "all"
                )
        )

    def _resize_search_content(
        self,
        event
    ):

        self.search_canvas.itemconfigure(
            self.search_window,
            width=event.width
        )

    def on_mousewheel(
        self,
        event
    ):

        try:

            if (
                self.notebook.select()
                !=
                str(self.tab_search)
            ):

                return

        except Exception:

            return

        self.search_canvas.yview_scroll(
            int(
                -1 *
                (event.delta / 120)
            ),
            "units"
        )

    # ========================================================
    # SEARCH EXECUTION
    # ========================================================

    def clear_results(self):

        for widget in (
            self.results_frame
            .winfo_children()
        ):

            widget.destroy()

    def search_images_ui(self):

        collection = (
            self.search_collection_var
            .get()
            .strip()
        )

        query = (
            self.query_entry
            .get()
            .strip()
        )

        if not collection:

            messagebox.showerror(
                "Ошибка",
                "Выберите коллекцию."
            )

            return

        if not query:

            messagebox.showerror(
                "Ошибка",
                "Введите поисковый запрос."
            )

            return

        try:

            manual_weight = float(
                self.manual_weight_spin.get()
            )

            auto_weight = float(
                self.auto_weight_spin.get()
            )

            top_k = int(
                self.top_k_spin.get()
            )

        except ValueError:

            messagebox.showerror(
                "Ошибка",
                "Проверьте параметры поиска."
            )

            return

        if (
            manual_weight < 0
            or auto_weight < 0
            or top_k < 1
        ):

            messagebox.showerror(
                "Ошибка",
                "Параметры поиска должны быть корректными."
            )

            return

        self.clear_results()

        try:

            results = search_images(

                query=query,

                collection_name=
                    collection,

                top_k=top_k,

                manual_weight=
                    manual_weight,

                auto_weight=
                    auto_weight
            )

            if not results:

                ttk.Label(
                    self.results_frame,
                    text=
                        "Ничего не найдено."
                ).pack(
                    anchor="w"
                )

                return

            for index, result in enumerate(
                results,
                start=1
            ):

                self.add_result(
                    index,
                    result
                )

            self._update_scroll_region()

        except Exception as e:

            messagebox.showerror(
                "Ошибка поиска",
                str(e)
            )

    # ========================================================
    # RESULT
    # ========================================================

    def add_result(
        self,
        index,
        result
    ):

        payload = result[
            "payload"
        ]

        final_score = result[
            "final_score"
        ]

        auto_score = result[
            "auto_score"
        ]

        manual_score = result[
            "manual_score"
        ]

        filename = payload.get(
            "filename",
            "неизвестный файл"
        )

        frame = ttk.LabelFrame(
            self.results_frame,
            text=
                f"Результат №{index}: "
                f"{filename}"
        )

        frame.pack(
            fill="x",
            pady=8
        )

        if auto_score is not None:

            auto_text = (
                f"{auto_score:.4f}"
            )

        else:

            auto_text = "нет"

        if manual_score is not None:

            manual_text = (
                f"{manual_score:.4f}"
            )

        else:

            manual_text = "нет"

        score_text = (

            f"Итоговый балл: "
            f"{final_score:.4f}\n"

            f"Автоматический балл: "
            f"{auto_text}\n"

            f"Ручной балл: "
            f"{manual_text}"
        )

        ttk.Label(
            frame,
            text=score_text,
            justify="left"
        ).pack(
            anchor="w",
            padx=10,
            pady=5
        )

        auto_description = (
            payload.get(
                "auto_description",
                ""
            )
        )

        ttk.Label(
            frame,
            text=(
                "Автоматическое описание:\n"
                +
                auto_description
            ),
            justify="left",
            wraplength=900
        ).pack(
            anchor="w",
            padx=10,
            pady=3
        )

        manual_description = (
            payload.get(
                "manual_description"
            )
        )

        if manual_description:

            ttk.Label(
                frame,
                text=(
                    "Ручная разметка:\n"
                    +
                    manual_description
                ),
                justify="left",
                wraplength=900
            ).pack(
                anchor="w",
                padx=10,
                pady=3
            )

        s3_key = payload.get(
            "s3_key"
        )

        if s3_key:

            try:

                image = (
                    load_image_from_s3(
                        s3_key
                    )
                )

                image.thumbnail(
                    (250, 250)
                )

                image_tk = (
                    ImageTk.PhotoImage(
                        image
                    )
                )

                image_label = ttk.Label(
                    frame,
                    image=image_tk
                )

                image_label.image = (
                    image_tk
                )

                image_label.pack(
                    anchor="w",
                    padx=10,
                    pady=5
                )

            except Exception as e:

                ttk.Label(
                    frame,
                    text=(
                        "Ошибка загрузки "
                        f"изображения: {e}"
                    )
                ).pack(
                    anchor="w",
                    padx=10
                )