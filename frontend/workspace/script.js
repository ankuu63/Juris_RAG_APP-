let selectedPDF = null;
let deleteTargetPDF = null;


// =========================
// LOAD PDFs
// =========================

async function loadPDFs() {

    const response = await fetch('/pdfs');

    const data = await response.json();

    const pdfList = document.getElementById('pdfList');

    pdfList.innerHTML = '';


    data.pdfs.forEach(pdf => {

        const card = document.createElement('div');

        card.className = 'pdf-card';


        card.innerHTML = `

            <div class="pdf-left">

                <img src="/static/workspace/assets/pdf-icon.png">

                <div class="pdf-name">
                    ${pdf}
                </div>

            </div>


            <div class="pdf-actions">

                <button class="icon-btn"
                        onclick="loadChatHistory('${pdf}')">

                    <img src="/static/workspace/assets/chat-icon.png">

                </button>


                <button class="icon-btn"
                        onclick="openDeleteModal('${pdf}')">

                    <img src="/static/workspace/assets/delete-icon.png">

                </button>

            </div>

        `;


        pdfList.appendChild(card);

    });
}


// =========================
// CHAT SUBMIT
// =========================

const chatForm = document.getElementById('chatForm');

chatForm.addEventListener('submit', async (e) => {

    e.preventDefault();


    const queryInput = document.getElementById('query');

    const pdfInput = document.getElementById('pdf');

    const chatArea = document.getElementById('chatArea');


    const query = queryInput.value.trim();

    if (!query) return;


    // =========================
    // USER MESSAGE
    // =========================

    const userMessage = document.createElement('div');

    userMessage.className = 'user-message';

    userMessage.innerText = query;

    chatArea.appendChild(userMessage);


    chatArea.scrollTop = chatArea.scrollHeight;


    // =========================
    // FORM DATA
    // =========================

    const formData = new FormData();

    formData.append('query', query);


    if (pdfInput.files.length > 0) {

        formData.append('pdf', pdfInput.files[0]);

        selectedPDF = pdfInput.files[0].name;
    }


    if (selectedPDF) {

        formData.append('selected_pdf', selectedPDF);
    }


    queryInput.value = '';


    // =========================
    // FETCH CHAT RESPONSE
    // =========================

    const response = await fetch('/chat', {

        method: 'POST',

        body: formData
    });


    const data = await response.json();


    // =========================
    // BOT MESSAGE
    // =========================

    const botMessage = document.createElement('div');

    botMessage.className = 'bot-message';


    if (data.error) {

        botMessage.innerText = data.error;

    } else {

        botMessage.innerText = data.answer;
    }


    chatArea.appendChild(botMessage);


    chatArea.scrollTop = chatArea.scrollHeight;


    // =========================
    // RELOAD PDF LIST
    // =========================

    loadPDFs();


    pdfInput.value = '';
});


// =========================
// LOAD CHAT HISTORY
// =========================

async function loadChatHistory(pdfName) {

    selectedPDF = pdfName;


    const response = await fetch(`/chat-history/${pdfName}`);

    const data = await response.json();


    const chatArea = document.getElementById('chatArea');

    chatArea.innerHTML = '';


    const welcome = document.createElement('div');

    welcome.className = 'bot-message';

    welcome.innerText = `Chatting with ${pdfName}`;

    chatArea.appendChild(welcome);


    data.messages.forEach(msg => {

        const messageDiv = document.createElement('div');


        if (msg.sender === 'user') {

            messageDiv.className = 'user-message';

        } else {

            messageDiv.className = 'bot-message';
        }


        messageDiv.innerText = msg.message;

        chatArea.appendChild(messageDiv);
    });


    chatArea.scrollTop = chatArea.scrollHeight;
}


// =========================
// DELETE MODAL
// =========================

function openDeleteModal(pdfName) {

    deleteTargetPDF = pdfName;

    document.getElementById('deleteModal').style.display = 'flex';

    document.getElementById('deleteMessage').innerText =
        `Delete ${pdfName}?`;
}


function closeDeleteModal() {

    document.getElementById('deleteModal').style.display = 'none';
}


// =========================
// DELETE PDF
// =========================

async function confirmDeletePDF() {

    if (!deleteTargetPDF) return;


    await fetch(`/delete-pdf/${deleteTargetPDF}`, {

        method: 'DELETE'
    });


    closeDeleteModal();


    if (selectedPDF === deleteTargetPDF) {

        selectedPDF = null;

        document.getElementById('chatArea').innerHTML = `
            <div class="bot-message">
                Welcome to JurisRAG 🚀
            </div>
        `;
    }


    loadPDFs();
}


// =========================
// INITIAL LOAD
// =========================

loadPDFs();