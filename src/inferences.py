from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load model in 4bit
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path="Qwen/Qwen3-4B-Thinking-2507", load_in_4bit=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="Qwen/Qwen3-4B-Thinking-2507")


with torch.no_grad():
    inputs = [{"role":"system","content":"Bạn là một trợ lí Tiếng Việt nhiệt tình và trung thực. Hãy luôn trả lời một cách hữu ích nhất có thể và vui lòng đặt đáp án cuối cùng của bạn trong \\boxed{}."},
              {"role":"user","content":"Chú ý các yêu cầu sau:\n- Câu trả lời phải chính xác và đầy đủ nếu ngữ cảnh có câu trả lời. \n- Chỉ sử dụng các thông tin có trong ngữ cảnh được cung cấp.\n- Chỉ cần từ chối trả lời và không suy luận gì thêm nếu ngữ cảnh không có câu trả lời.\nHãy trả lời câu hỏi dựa trên ngữ cảnh:\n### Ngữ cảnh :\nJ2SE hay Java 2 Standard Edition  vừa là một đặc tả, cũng vừa là một nền tảng thực thi (bao gồm cả phát triển và triển khai) cho các ứng dụng Java. Nó cung cấp các API, các kiến trúc chuẩn, các thư viện lớp và các công cụ cốt lõi nhất để xây các ứng dụng Java. Mặc dù J2SE là nền tảng thiên về phát triển các sản phẩm chạy trên máy tính để bàn nhưng những tính năng của nó, bao gồm phần triển khai ngôn ngữ Java lớp gốc, các công nghệ nền như JDBC để truy vấn dữ liệu... chính là chỗ dựa để Java tiếp tục mở rộng và hỗ trợ các thành phần mạnh mẽ hơn dùng cho các ứng dụng hệ thống quy mô xí nghiệp và các thiết bị nhỏ.\nJ2SE gồm 2 bộ phận chính là:\nMôi trường thực thi hay JRE cung cấp các Java API, máy ảo Java (Java Virtual Machine hay JVM) và các thành phần cần thiết khác để chạy các applet và ứng dụng viết bằng ngôn ngữ lập trình Java. Môi trường thực thi Java không có các công cụ và tiện ích như là các trình biên dịch hay các trình gỡ lỗi để phát triển các applet và các ứng dụng.\nJava 2 SDK là một tập mẹ của JRE, và chứa mọi thứ nằm trong JRE, bổ sung thêm các công cụ như là trình biên dịch (compiler) và các trình gỡ lỗi (debugger) cần để phát triển applet và các ứng dụng.\nTên J2SE (Java 2 Platform, Standard Edition) được sử dụng từ phiên bản 1.2 cho đến 1.5. Từ \"SE\" được sử dụng để phân biệt với các nền tảng khác là Java EE và Java ME. \"2\" ban đầu vốn được dùng để chỉ đến những thay đổi lớn trong phiên bản 1.2 so với các phiên bản trước, nhưng đến phiên bản 1.6 thì \"2\" bị loại bỏ.\nPhiên bản được biết đến tới thời điểm hiện tại là Java SE 6 (hay Java SE 1.6 theo cách đặt tên của Sun Microsystems) với tên mã Mustang.\n\n### Câu hỏi :\nDựa trên thông tin từ đoạn văn, hãy giải thích tại sao J2SE được coi là nền tảng quan trọng trong việc hỗ trợ các ứng dụng Java, đặc biệt là các ứng dụng hệ thống quy mô xí nghiệp và các thiết bị nhỏ?\n\n\"Hãy suy nghĩ một cách tuần tự và logic từng bước một để giải quyết câu hỏi truy vấn và chỉ đưa ra đáp án cuối cùng sau khi đã suy luận đầy đủ.\"\n"}]
    text = tokenizer.apply_chat_template(
        inputs,
        tokenize=False,
        add_generation_prompt=True,
        # enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=1000,
        top_p=0.3,
    )
    output_ids = generated_ids[0][len(inputs.input_ids[0]):].tolist()

    # # decode output
    # output_text = tokenizer.decode(output_ids, skip_special_tokens=False)
    # print(output_text)

    try:
    # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    print("thinking content:", thinking_content) # no opening <think> tag
    print("content:", content)



    # streaming generation
