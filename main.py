import tkinter
import customtkinter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

vectorizer = TfidfVectorizer()

customtkinter.set_appearance_mode("System")
customtkinter.set_default_color_theme("blue")


class InputBox(customtkinter.CTkFrame):
    def __init__(self, master, title, **kwargs):
        super().__init__(master, **kwargs)

        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.title = customtkinter.CTkLabel(master=self, text=title, font=("Arial", 20))
        self.title.grid(row=0, column=0, columnspan=3, sticky="nsew", padx=12, pady=12)

        self.textbox = customtkinter.CTkTextbox(master=self, width=500, corner_radius=0, font=("monospace", 12))
        self.textbox.grid(row=1, column=0, columnspan=3, sticky="nsew")

        self.textbox.bind("<KeyRelease>", self.on_key_release)
        self.textbox.bind("<ButtonRelease>", self.on_key_release)

        self.position = customtkinter.CTkLabel(master=self, text="Ln 0, Col 0", font=("Arial", 12))
        self.position.grid(row=2, column=0, sticky="nsw", padx=6)
        self.char_count = customtkinter.CTkLabel(master=self, text="0 chars", font=("Arial", 12))
        self.char_count.grid(row=2, column=1, sticky="nsew", padx=6)
        self.word_count = customtkinter.CTkLabel(master=self, text="0 words", font=("Arial", 12))
        self.word_count.grid(row=2, column=2, sticky="nsew", padx=6)

    def on_key_release(self, event):
        self.master.update_bar()
        self.update_status()

    def update_status(self):
        row, col = self.textbox.index('insert').split('.')
        self.position.configure(text=f'Ln {row}, Col {col}')
        self.char_count.configure(text=f'{len(self.get_text().strip())} chars')
        self.word_count.configure(text=f'{len(self.get_text().split())} words')
        self.after(100, self.update_status)

    def get_text(self):
        return self.textbox.get("0.0", "end")


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.geometry("1000x600")
        self.title("Plagiarism Checker")
        self.minsize(1000, 600)

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(2, weight=1)

        self.input1 = InputBox(master=self, title="Input 1")
        self.input1.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)

        self.meter = customtkinter.CTkProgressBar(master=self, orientation="vertical")
        self.meter.grid(row=0, column=1, sticky="ns", padx=0, pady=12)

        self.input2 = InputBox(master=self, title="Input 2")
        self.input2.grid(row=0, column=2, sticky="nsew", padx=12, pady=12)

        self.update_bar()

    def update_bar(self):
        text1 = self.input1.get_text().strip()
        text2 = self.input2.get_text().strip()
        if text1 == text2:
            self.meter.set(1)
        else:
            vectors = vectorizer.fit_transform([text1, text2])
            similarity_matrix = cosine_similarity(vectors)
            self.meter.set(similarity_matrix[0][1])


app = App()
app.mainloop()
