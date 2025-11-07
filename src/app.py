import gradio as gr
from json_map import search_alarms_by_name, low_urgency, medium_urgency, high_urgency

with gr.Blocks() as demo:
    gr.Markdown("# Alarm Monitor and Agent Summarizer")
    
    gr.Markdown("### 1. Search for Alarms")
    with gr.Row():
        alarm_name_input = gr.Textbox(label="Alarm Name")
        urgency_input = gr.Textbox(label="Urgency (optional)")
    
    search_button = gr.Button("Search")

    search_output = gr.JSON(label="Matching Alarms (Raw JSON)")

    search_button.click(
        fn=search_alarms_by_name,
        inputs=[alarm_name_input, urgency_input],
        outputs=search_output
    )
    
    gr.Markdown("---")
    gr.Markdown("### 2. Get Agent Summaries")
    gr.Markdown("After searching, click a button below to summarize the raw JSON results.")

    with gr.Row():
        low_btn = gr.Button("Summarize Low Urgency")
        med_btn = gr.Button("Summarize Medium Urgency")
        high_btn = gr.Button("Summarize High Urgency")
    

    summary_output_json = gr.JSON(
        label="Agent Summary JSON"
    )
    low_btn.click(
        fn=low_urgency,
        inputs=[search_output],
        outputs=summary_output_json 
    )
    
    med_btn.click(
        fn=medium_urgency,
        inputs=[search_output],
        outputs=summary_output_json 
    )
    
    high_btn.click(
        fn=high_urgency,
        inputs=[search_output],
        outputs=summary_output_json 
    )

demo.launch(share=True)