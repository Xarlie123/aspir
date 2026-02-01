# ui/utils/widget_helpers.py
def embed_widget(widget, placeholder, layout):
    widget.setSizePolicy(placeholder.sizePolicy())
    widget.setMinimumSize(placeholder.minimumSize())
    widget.setMaximumSize(placeholder.maximumSize())
    placeholder.hide()
    layout.replaceWidget(placeholder, widget)
    widget.show()
