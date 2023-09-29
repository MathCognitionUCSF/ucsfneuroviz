import ipywidgets as widgets
from IPython.display import display, HTML


def generate_legend():
    legend_content = """
    <div>
        <div style="display: flex; flex-direction: column;">
            <div><strong>Legend</strong></div>
            <div><span style="background-color: #FF5733; padding: 2px 10px;"></span> Abbreviation 1</div>
            <div><span style="background-color: #33FF57; padding: 2px 10px;"></span> Abbreviation 2</div>
            <!-- Add more entries as needed -->
        </div>
    </div>
    """
    
    legend_widget = HTML(legend_content)
    
    return legend_widget

# Create a legend widget
legend = generate_legend()
