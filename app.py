import os
import uuid
import math
import shutil  # Dùng để xóa thư mục
import streamlit as st
import chromadb
from PIL import Image
from sentence_transformers import SentenceTransformer
import easyocr
import numpy as np
from streamlit_paste_button import paste_image_button

# Tắt các cảnh báo không cần thiết từ Hugging Face & PyTorch
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Cấu hình trang
st.set_page_config(
    page_title="Hệ thống Tra cứu Bao bì",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS nâng cấp giao diện mượt mà & hiện đại
st.markdown("""
<style>
    .stApp {
        background-color: #F8FAFC;
    }
    .main-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        padding: 20px 28px;
        border-radius: 16px;
        color: #FFFFFF;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.15);
        margin-bottom: 20px;
        text-align: center;
    }
    .main-header h1 {
        color: #FFFFFF !important;
        font-weight: 700;
        font-size: 1.7rem;
        margin: 0;
    }
    .main-header p {
        color: #94A3B8;
        font-size: 0.9rem;
        margin-top: 4px;
        margin-bottom: 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #E2E8F0;
        padding: 4px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        border-radius: 8px;
        background-color: transparent;
        border: none;
        color: #475569;
        font-weight: 600;
        padding: 0px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        border: none;
        border-radius: 8px;
        font-weight: 600;
        box-shadow: 0 4px 10px rgba(37, 99, 235, 0.2);
    }
</style>
""", unsafe_allow_html=True)

CHROMA_DB_DIR = "./chroma_db"
IMAGE_STORAGE_DIR = "./registered_images"
os.makedirs(IMAGE_STORAGE_DIR, exist_ok=True)

# Tải Model tối ưu với caching
@st.cache_resource(show_spinner=False)
def load_clip_model():
    return SentenceTransformer('clip-ViT-B-32')

@st.cache_resource(show_spinner=False)
def load_ocr_reader():
    return easyocr.Reader(['vi', 'en'], gpu=False, quantize=True)

@st.cache_resource(show_spinner=False)
def init_vector_db():
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    return client.get_or_create_collection(
        name="food_products",
        metadata={"hnsw:space": "cosine"}
    )

def optimize_image_for_ai(image: Image.Image, max_size=640) -> Image.Image:
    img = image.convert('RGB')
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return img

def get_image_embedding(image: Image.Image):
    model = load_clip_model()
    opt_img = optimize_image_for_ai(image, max_size=512)
    return model.encode(opt_img).tolist()

def extract_text_from_image(image: Image.Image):
    reader = load_ocr_reader()
    opt_img = optimize_image_for_ai(image, max_size=800)
    img_np = np.array(opt_img)
    results = reader.readtext(img_np, detail=0)
    return " ".join(results).upper()

# Banner Header
st.markdown("""
<div class="main-header">
    <h1>🔍 HỆ THỐNG TRA CỨU BAO BÌ</h1>
    <p>Nhận diện thông minh & Trích xuất thông tin bao bì tự động</p>
</div>
""", unsafe_allow_html=True)

collection = init_vector_db()

tab1, tab2, tab3 = st.tabs(["🔎 Tra Cứu Hình Ảnh", "➕ Đăng Ký Bao Bì Mới", "📋 Danh Sách & Quản Lý"])

# TAB 1: TRA CỨU BAO BÌ
with tab1:
    col1, col2 = st.columns([1, 1], gap="large")
    
    if "search_paste_key" not in st.session_state:
        st.session_state.search_paste_key = 0
    if "search_image" not in st.session_state:
        st.session_state.search_image = None
        
    with col1:
        with st.container(border=True):
            st.subheader("1. Tải lên, Chụp ảnh hoặc Dán ảnh")
            source_option = st.radio(
                "Nguồn ảnh:", 
                ["Tải file lên (Upload)", "Dán ảnh từ Clipboard (Copy trên máy tính)", "Chụp từ Camera"], 
                horizontal=True,
                key="search_source_opt"
            )
            
            if source_option == "Tải file lên (Upload)":
                uploaded_file = st.file_uploader("Kéo thả hoặc chọn file ảnh bao bì:", type=["jpg", "jpeg", "png", "webp"], key=f"search_uploader_{st.session_state.search_paste_key}")
                if uploaded_file:
                    st.session_state.search_image = Image.open(uploaded_file)
                    
            elif source_option == "Dán ảnh từ Clipboard (Copy trên máy tính)":
                st.write("Sao chép (Copy) file ảnh hoặc ảnh từ màn hình, sau đó bấm nút:")
                btn_col_a, btn_col_b = st.columns([3, 2])
                with btn_col_a:
                    paste_result = paste_image_button(
                        label="📋 Click vào đây để Dán ảnh đã Copy",
                        background_color="#2563EB",
                        text_color="#FFFFFF",
                        hover_background_color="#1D4ED8",
                        key=f"paste_search_btn_{st.session_state.search_paste_key}"
                    )
                    if paste_result.image_data is not None:
                        st.session_state.search_image = paste_result.image_data
                with btn_col_b:
                    if st.button("❌ Xóa ảnh", key="clear_search_img", use_container_width=True):
                        st.session_state.search_image = None
                        st.session_state.search_paste_key += 1
                        st.rerun()
                    
            else:
                camera_file = st.camera_input("Chụp ảnh trực tiếp bao bì:", key=f"search_camera_{st.session_state.search_paste_key}")
                if camera_file:
                    st.session_state.search_image = Image.open(camera_file)
            
            if st.session_state.search_image is not None:
                st.image(st.session_state.search_image, caption="Ảnh bao bì hiện tại", use_container_width=True)
                threshold = st.slider("Ngưỡng tin cậy hình ảnh (%)", min_value=50, max_value=95, value=70, step=5) / 100.0
                search_btn = st.button("🚀 Bắt đầu Tra Cứu", type="primary", use_container_width=True)

    with col2:
        with st.container(border=True):
            st.subheader("2. Kết quả nhận diện")
            if st.session_state.search_image is not None and 'search_btn' in locals() and search_btn:
                with st.spinner("⚡ AI đang xử lý siêu tốc..."):
                    detected_text = extract_text_from_image(st.session_state.search_image)
                    query_vector = get_image_embedding(st.session_state.search_image)
                    results = collection.query(query_embeddings=[query_vector], n_results=5)
                    
                    if not results['ids'] or not results['ids'][0]:
                        st.warning("⚠️ Cơ sở dữ liệu chưa có bao bì nào.")
                    else:
                        best_match = None
                        highest_score = -1.0
                        
                        for idx in range(len(results['ids'][0])):
                            doc_id = results['ids'][0][idx]
                            metadata = results['metadatas'][0][idx]
                            distance = results['distances'][0][idx]
                            similarity = 1.0 - distance
                            
                            p_id = str(metadata.get('product_id', '')).upper()
                            p_name = str(metadata.get('product_name', '')).upper()
                            
                            text_matched = False
                            if (p_id and p_id in detected_text) or (p_name and p_name in detected_text):
                                text_matched = True
                            
                            final_score = similarity + (0.3 if text_matched else 0.0)
                            
                            if final_score > highest_score:
                                highest_score = final_score
                                best_match = {
                                    "doc_id": doc_id,
                                    "metadata": metadata,
                                    "similarity": similarity,
                                    "text_matched": text_matched
                                }
                        
                        if best_match and (best_match['similarity'] >= threshold or best_match['text_matched']):
                            meta = best_match['metadata']
                            st.success("✅ **ĐÃ XÁC NHẬN CHÍNH XÁC BAO BÌ**")
                            if best_match['text_matched']:
                                st.info("💡 **Khớp thành công chữ in/mã hàng trên bao bì!**")
                            
                            st.markdown(f"""
                            * **Mã Bao Bì:** `{meta.get('product_id', best_match['doc_id'])}`
                            * **Tên Bao Bì:** **{meta.get('product_name', 'Không có')}**
                            * **Khách Hàng:** **{meta.get('customer_name', 'Không có')}**
                            * **Dạng Bao Bì:** **{meta.get('package_type', 'Không có')}**
                            * **Tiêu Chuẩn Đóng Gói:** {meta.get('packaging_spec', 'Không có')}
                            """)
                            
                            st.caption(f"Chữ AI đọc trên ảnh: `{detected_text}`")
                            
                            registered_img_path = meta.get('image_path')
                            if registered_img_path and os.path.exists(registered_img_path):
                                st.image(registered_img_path, caption=f"Ảnh mẫu ({meta.get('product_name', '')})", use_container_width=True)
                        else:
                            st.error("❌ **KHÔNG TÌM THẤY BAO BÌ PHÙ HỢP**")
                            st.write(f"Chữ AI đọc được từ ảnh: `{detected_text}`")

# TAB 2: ĐĂNG KÝ BAO BÌ MỚI
with tab2:
    with st.container(border=True):
        st.subheader("Đăng Ký Bao Bì Mẫu Mới")
        
        if "reg_paste_key" not in st.session_state:
            st.session_state.reg_paste_key = 0
        if "reg_image" not in st.session_state:
            st.session_state.reg_image = None

        col_a, col_b = st.columns(2, gap="medium")
        
        with col_a:
            product_id = st.text_input("Mã bao bì / Item Code *", placeholder="VD: 1748, 1749, T1851...")
            product_name = st.text_input("Tên bao bì *", placeholder="VD: BÁNH TÉT CHUỐI...")
            customer_name = st.text_input("Khách hàng *", placeholder="VD: LB, Duy Khoi Foods...")
            package_type = st.text_input("Dạng bao bì *", placeholder="VD: Thùng, Túi PE 500g...")
            packaging_spec = st.text_area("Tiêu chuẩn đóng gói *", placeholder="VD: Đóng gói 6 kg/thùng...")

        with col_b:
            st.write("**Tải lên ảnh mẫu bao bì ***")
            img_source_reg = st.radio(
                "Cách nhập ảnh mẫu:", 
                ["Tải file lên (Upload)", "Dán ảnh từ Clipboard (Copy trên máy tính)"], 
                horizontal=True,
                key="reg_img_source"
            )
            
            if img_source_reg == "Tải file lên (Upload)":
                uploaded_reg = st.file_uploader("Kéo thả hoặc chọn file ảnh bao bì mẫu:", type=["jpg", "jpeg", "png", "webp"], key=f"reg_uploader_{st.session_state.reg_paste_key}")
                if uploaded_reg:
                    st.session_state.reg_image = Image.open(uploaded_reg)
                if st.session_state.reg_image is not None:
                    if st.button("❌ Xóa ảnh", key="clear_reg_upload_img"):
                        st.session_state.reg_image = None
                        st.session_state.reg_paste_key += 1
                        st.rerun()
            else:
                st.write("Sao chép (Copy) ảnh từ máy tính, sau đó nhấn nút:")
                paste_col, clear_col = st.columns([3, 2])
                with paste_col:
                    paste_reg_btn = paste_image_button(
                        label="📋 Click vào đây để Dán ảnh đã Copy",
                        background_color="#2563EB",
                        text_color="#FFFFFF",
                        hover_background_color="#1D4ED8",
                        key=f"paste_reg_btn_{st.session_state.reg_paste_key}"
                    )
                    if paste_reg_btn.image_data is not None:
                        st.session_state.reg_image = paste_reg_btn.image_data
                with clear_col:
                    if st.button("❌ Xóa ảnh", key="clear_reg_img", use_container_width=True):
                        st.session_state.reg_image = None
                        st.session_state.reg_paste_key += 1
                        st.rerun()

            if st.session_state.reg_image is not None:
                st.image(st.session_state.reg_image, caption="Ảnh mẫu chuẩn bị lưu", use_container_width=True)

        submitted = st.button("💾 Lưu Bao Bì Vào Database", type="primary", use_container_width=True)
        if submitted:
            if not product_id or not product_name or not customer_name or not package_type or not packaging_spec or st.session_state.reg_image is None:
                st.error("Vui lòng điền đầy đủ thông tin có dấu (*) và chọn/dán ảnh mẫu!")
            else:
                try:
                    db_id = product_id.strip()
                    saved_img_filename = f"{db_id}_{customer_name}.png".replace(" ", "_").replace("/", "_")
                    saved_img_path = os.path.join(IMAGE_STORAGE_DIR, saved_img_filename)
                    
                    sample_img = optimize_image_for_ai(st.session_state.reg_image, max_size=1024)
                    sample_img.convert('RGB').save(saved_img_path)
                    
                    vector = get_image_embedding(sample_img)
                    collection.add(
                        ids=[db_id],
                        embeddings=[vector],
                        metadatas=[{
                            "product_id": db_id,
                            "customer_name": customer_name,
                            "product_name": product_name,
                            "package_type": package_type,
                            "packaging_spec": packaging_spec,
                            "image_path": saved_img_path
                        }]
                    )
                    st.session_state.reg_image = None
                    st.session_state.reg_paste_key += 1
                    st.success(f"🎉 Đã đăng ký thành công bao bì **{product_name}** (Mã: `{db_id}`)!")
                except Exception as e:
                    st.error(f"Lỗi khi lưu dữ liệu (Mã bao bì có thể đã tồn tại): {e}")

# TAB 3: DANH SÁCH & QUẢN LÝ (ĐÃ THÊM NÚT RESET DATABASE CHUYÊN DỤNG)
with tab3:
    st.header("Quản lý Cơ sở dữ liệu Bao bì")
    
    # Khu vực Admin đặc biệt để Reset dữ liệu
    with st.expander("⚡ KHU VỰC QUẢN TRỊ (CHỈ DÙNG KHI CẦN RESET)"):
        st.error("⚠️ HÀNH ĐỘNG NÀY SẼ XÓA TOÀN BỘ DỮ LIỆU ĐANG CÓ TRÊN CLOUD!")
        st.markdown("""
        * Toàn bộ ảnh mẫu đã nạp sẽ bị xóa.
        * Toàn bộ database tra cứu vector sẽ bị xóa về trạng thái trống.
        * Hãy chắc chắn bạn đã có bản sao dữ liệu trước khi thực hiện.
        """)
        
        # Thêm biến xác nhận để tránh bấm nhầm
        confirm_reset = st.checkbox("Tôi xác nhận muốn xóa toàn bộ database (Không thể hoàn tác)")
        if confirm_reset:
            reset_db_btn = st.button("🔥 CHẤP NHẬN RESET TOÀN BỘ DATABASE", type="secondary", use_container_width=True)
            if reset_db_btn:
                with st.spinner("Đang thực hiện Reset database... Vui lòng không đóng trang web."):
                    try:
                        # 1. Xóa Database Chroma
                        if os.path.exists(CHROMA_DB_DIR):
                            shutil.rmtree(CHROMA_DB_DIR)
                        
                        # 2. Xóa Thư mục ảnh mẫu
                        if os.path.exists(IMAGE_STORAGE_DIR):
                            shutil.rmtree(IMAGE_STORAGE_DIR)
                        
                        st.success("🎉 Reset hoàn tất! Hệ thống đã trống dữ liệu.")
                        st.warning("🔄 Trang web sẽ tự động tải lại sau 3 giây.")
                        # Tải lại trang sau khi xóa
                        st.cache_resource.clear()
                        import time
                        time.sleep(3)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Lỗi khi Reset database: {e}")

    # Hiển thị danh sách như bình thường
    st.markdown("---")
    st.subheader("📋 Danh sách bao bì đang lưu trữ")
    
    all_data = collection.get()
    
    if not all_data['ids']:
        st.info("Chưa có bao bì nào được đăng ký.")
    else:
        search_kw = st.text_input("🔍 Tìm kiếm nhanh bao bì:", placeholder="Nhập tên, mã bao bì hoặc tên khách hàng...").strip().lower()
        
        filtered_items = []
        for idx, item_id in enumerate(all_data['ids']):
            metadata = all_data['metadatas'][idx]
            p_id = str(metadata.get('product_id', item_id)).lower()
            p_name = str(metadata.get('product_name', '')).lower()
            c_name = str(metadata.get('customer_name', '')).lower()
            
            if not search_kw or (search_kw in p_id or search_kw in p_name or search_kw in c_name):
                filtered_items.append((item_id, metadata))
                
        total_items = len(filtered_items)
        st.caption(f"Hiển thị **{total_items}** / {len(all_data['ids'])} bao bì")
        
        if total_items == 0:
            st.warning("Không tìm thấy bao bì nào phù hợp.")
        else:
            ITEMS_PER_PAGE = 12
            total_pages = math.ceil(total_items / ITEMS_PER_PAGE)
            
            page_col1, page_col2 = st.columns([1, 4])
            with page_col1:
                current_page = st.number_input("Trang:", min_value=1, max_value=max(1, total_pages), value=1, step=1)
            with page_col2:
                st.write(f"<br>Trang **{current_page}** / **{total_pages}**", unsafe_allow_html=True)
                
            start_idx = (current_page - 1) * ITEMS_PER_PAGE
            end_idx = start_idx + ITEMS_PER_PAGE
            page_items = filtered_items[start_idx:end_idx]

            cols = st.columns(2)
            if "edit_id" not in st.session_state:
                st.session_state.edit_id = None

            for idx, (item_id, metadata) in enumerate(page_items):
                with cols[idx % 2]:
                    with st.container(border=True):
                        img_path = metadata.get('image_path')
                        if img_path and os.path.exists(img_path):
                            st.image(img_path, use_container_width=True)
                        st.subheader(metadata.get('product_name', 'Bao bì không tên'))
                        st.caption(f"🆔 **Mã Bao Bì:** `{metadata.get('product_id', item_id)}`")
                        st.write(f"🏢 **Khách hàng:** {metadata.get('customer_name', 'Không có')}")
                        st.write(f"📦 **Dạng bao bì:** {metadata.get('package_type', 'Không có')}")
                        st.write(f"📋 **Tiêu chuẩn đóng gói:** {metadata.get('packaging_spec', 'Không có')}")
                        
                        btn_col1, btn_col2 = st.columns(2)
                        with btn_col1:
                            if st.button(f"✏️ Chỉnh sửa", key=f"edit_btn_{item_id}", use_container_width=True):
                                st.session_state.edit_id = item_id if st.session_state.edit_id != item_id else None
                                st.rerun()
                        with btn_col2:
                            if st.button(f"🗑️ Xóa bao bì", key=f"del_btn_{item_id}", type="secondary", use_container_width=True):
                                collection.delete(ids=[item_id])
                                if img_path and os.path.exists(img_path):
                                    try:
                                        os.remove(img_path)
                                    except Exception:
                                        pass
                                st.success("Đã xóa bao bì thành công!")
                                st.rerun()

                        if st.session_state.edit_id == item_id:
                            st.markdown("---")
                            st.write("##### 📝 Cập nhật thông tin bao bì:")
                            with st.form(key=f"edit_form_{item_id}"):
                                new_prod_id = st.text_input("Mã bao bì", value=metadata.get('product_id', item_id))
                                new_prod_name = st.text_input("Tên bao bì", value=metadata.get('product_name', ''))
                                new_cust_name = st.text_input("Khách hàng", value=metadata.get('customer_name', ''))
                                new_pkg_type = st.text_input("Dạng bao bì", value=metadata.get('package_type', ''))
                                new_pkg_spec = st.text_area("Tiêu chuẩn đóng gói", value=metadata.get('packaging_spec', ''))
                                new_img_file = st.file_uploader("Thay ảnh mẫu mới", type=["jpg", "jpeg", "png", "webp"])
                                
                                c1, c2 = st.columns(2)
                                with c1:
                                    save_edit = st.form_submit_button("💾 Lưu thay đổi", type="primary", use_container_width=True)
                                with c2:
                                    cancel_edit = st.form_submit_button("❌ Hủy", use_container_width=True)

                                if save_edit:
                                    current_img_path = metadata.get('image_path')
                                    vector = None
                                    
                                    if new_img_file:
                                        sample_img = optimize_image_for_ai(Image.open(new_img_file), max_size=1024)
                                        vector = get_image_embedding(sample_img)
                                        
                                        file_ext = os.path.splitext(new_img_file.name)[1]
                                        saved_img_filename = f"{new_prod_id}_{new_cust_name}{file_ext}".replace(" ", "_").replace("/", "_")
                                        current_img_path = os.path.join(IMAGE_STORAGE_DIR, saved_img_filename)
                                        sample_img.convert('RGB').save(current_img_path)

                                    updated_metadata = {
                                        "product_id": new_prod_id,
                                        "customer_name": new_cust_name,
                                        "product_name": new_prod_name,
                                        "package_type": new_pkg_type,
                                        "packaging_spec": new_pkg_spec,
                                        "image_path": current_img_path
                                    }

                                    if vector:
                                        collection.update(ids=[item_id], embeddings=[vector], metadatas=[updated_metadata])
                                    else:
                                        collection.update(ids=[item_id], metadatas=[updated_metadata])

                                    st.session_state.edit_id = None
                                    st.success("Cập nhật thông tin thành công!")
                                    st.rerun()

                                if cancel_edit:
                                    st.session_state.edit_id = None
                                    st.rerun()
