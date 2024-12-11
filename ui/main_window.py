import tkinter as tk
from tkinter import filedialog

def main_window():
    root = tk.Tk()
    root.title("Math Formula Drawer")
    root.geometry("800x600")

    canvas = tk.Canvas(root, bg="white", width=600, height=400)
    canvas.pack(pady=20)

    def clear_canvas():
        canvas.delete("all")

    clear_button = tk.Button(root, text="Clear Canvas", command=clear_canvas)
    clear_button.pack()

    root.mainloop()

if __name__ == "__main__":
    main_window()
