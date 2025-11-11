import gradio as gr
from json_map import low_urgency, medium_urgency, high_urgency
from agentic_mapping import generate_final_output

custom_css = """
/* Global font size */
body, .gradio-container {
    font-size: 20px !important;
}

/* Markdown title and descriptions */
.gr-markdown {
    font-size: 22px !important;
    line-height: 1.6 !important;
}

#summary_output .prose, 
#summary_output .prose * {
    font-size: 30px !important;
    line-height: 1.8 !important;
}

/* JSON output */
.gr-json {
    font-size: 18px !important;
}

/* Buttons */
button {
    font-size: 20px !important;
    padding: 8px 14px !important;
}

/* Text input font size */
input, textarea {
    font-size: 20px !important;
}
"""


with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("# Alarm Monitor and Agent Summarizer")
    
    gr.Markdown("### 1. Search for Alarms")
    with gr.Row():
        alarm_name_input = gr.Textbox(label="Alarm Name")
        urgency_input = gr.Textbox(label="Urgency (optional)")
    
    search_button = gr.Button("Search")

    search_output = gr.JSON(
        label="Matching Alarms (Raw JSON)", 
        # visible=False
    )

    search_button.click(
        fn=generate_final_output,
        inputs=[alarm_name_input, urgency_input],
        outputs=search_output,
        scroll_to_output=True 

    )
    
    gr.Markdown("---")
    gr.Markdown("### 2. Get Agent Summaries")
    gr.Markdown("After searching, click a button below to summarize the raw JSON results.")

    with gr.Row():
        low_btn = gr.Button("Summarize Low Urgency")
        med_btn = gr.Button("Summarize Medium Urgency")
        high_btn = gr.Button("Summarize High Urgency")

    summary_output_markdown = gr.Markdown(
        label="Agent Summary",
        elem_id="summary_output"
    )
    
    low_btn.click(
        fn=low_urgency,
        inputs=[search_output],
        outputs=summary_output_markdown,
        show_progress="full",
        scroll_to_output=True
    )
    
    med_btn.click(
        fn=medium_urgency,
        inputs=[search_output],
        outputs=summary_output_markdown,
        show_progress="full",
        scroll_to_output=True
    )
    
    high_btn.click(
        fn=high_urgency,
        inputs=[search_output],
        outputs=summary_output_markdown,
        show_progress="full",
        scroll_to_output=True
    )

demo.launch(share=True)