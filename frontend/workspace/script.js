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

                <img src="/static/workspace/assets/pdf-icon.jpg">

                <div class="pdf-name">
                    ${pdf}
                </div>

            </div>


            <div class="pdf-actions">

                <button class="icon-btn"
                        onclick="loadChatHistory('${pdf}')">

                    <img src="/static/workspace/assets/chat-icon.jpg">

                </button>


                <button class="icon-btn"
                        onclick="openDeleteModal('${pdf}')">

                    <img src="/static/workspace/assets/delete-icon.jpg">

                </button>

            </div>

        `;

        pdfList.appendChild(card);
    });
}


// =========================
// FILE PICKER
// =========================

const pdfInput = document.getElementById('pdf');

pdfInput.addEventListener('change', () => {

    const fileNameText =
        document.getElementById('selectedFileName');


    if (pdfInput.files.length > 0) {

        fileNameText.innerText =
            pdfInput.files[0].name;

    } else {

        fileNameText.innerText = '';
    }
});


// =========================
// CHAT SUBMIT
// =========================

const chatForm = document.getElementById('chatForm');

chatForm.addEventListener('submit', async (e) => {

    e.preventDefault();


    const queryInput =
        document.getElementById('query');

    const askButton =
        document.getElementById('askButton');

    const chatArea =
        document.getElementById('chatArea');


    const query = queryInput.value.trim();

    if (!query) return;


    askButton.disabled = true;

    askButton.innerText = 'Thinking...';


    const userMessage =
        document.createElement('div');

    userMessage.className = 'user-message';

    userMessage.innerText = query;

    chatArea.appendChild(userMessage);

    chatArea.scrollTop = chatArea.scrollHeight;


    const formData = new FormData();

    formData.append('query', query);


    if (pdfInput.files.length > 0) {

        formData.append(
            'pdf',
            pdfInput.files[0]
        );

        selectedPDF =
            pdfInput.files[0].name;
    }


    if (selectedPDF) {

        formData.append(
            'selected_pdf',
            selectedPDF
        );
    }


    queryInput.value = '';


    const currentPDFText =
        document.getElementById(
            'currentPDFText'
        );


    if (selectedPDF) {

        currentPDFText.innerText =
            `Chatting with ${selectedPDF}`;
    }


    try {

        const response = await fetch('/chat', {

            method: 'POST',

            body: formData
        });

        const data = await response.json();


        const botMessage =
            document.createElement('div');

        botMessage.className =
            'bot-message';


        if (data.error) {

            botMessage.innerText =
                data.error;

        } else {

            let finalText =
                data.answer;


            if (data.sources) {

                finalText +=
                    "\n\nSources:\n";


                if (
                    Array.isArray(data.sources)
                ) {

                    data.sources.forEach(
                        source => {

                        if (
                            typeof source ===
                            "object"
                        ) {

                            const page =
                                source.page ||
                                "Unknown";

                            const content =
                                source.content ||
                                source.text ||
                                source.source ||
                                "Citation";

                            finalText +=
                                `• Page ${page}: ${content}\n`;

                        } else {

                            finalText +=
                                `• ${source}\n`;
                        }
                    });

                } else {

                    finalText +=
                        `• ${data.sources}`;
                }
            }


            botMessage.innerText =
                finalText;
        }


        chatArea.appendChild(
            botMessage
        );

        chatArea.scrollTop =
            chatArea.scrollHeight;


        loadPDFs();

    } catch (error) {

        const errorMessage =
            document.createElement('div');

        errorMessage.className =
            'bot-message';

        errorMessage.innerText =
            'Something went wrong.';

        chatArea.appendChild(
            errorMessage
        );
    }


    askButton.disabled = false;

    askButton.innerText = 'Ask';


    pdfInput.value = '';

    document.getElementById(
        'selectedFileName'
    ).innerText = '';
});


// =========================
// LOAD CHAT HISTORY
// =========================

async function loadChatHistory(pdfName) {

    selectedPDF = pdfName;


    const response = await fetch(
        `/chat-history/${encodeURIComponent(pdfName)}`
    );

    const data = await response.json();


    const chatArea =
        document.getElementById(
            'chatArea'
        );

    chatArea.innerHTML = '';


    document.getElementById(
        'currentPDFText'
    ).innerText =
        `Chatting with ${pdfName}`;


    data.messages.forEach(msg => {

        const messageDiv =
            document.createElement('div');


        if (msg.sender === 'user') {

            messageDiv.className =
                'user-message';

        } else {

            messageDiv.className =
                'bot-message';
        }


        messageDiv.innerText =
            msg.message;

        chatArea.appendChild(
            messageDiv
        );
    });


    chatArea.scrollTop =
        chatArea.scrollHeight;
}


// =========================
// DELETE MODAL
// =========================

function openDeleteModal(pdfName) {

    deleteTargetPDF = pdfName;

    document.getElementById(
        'deleteModal'
    ).style.display = 'flex';


    document.getElementById(
        'deleteMessage'
    ).innerText =
        `Delete ${pdfName}?`;
}


function closeDeleteModal() {

    document.getElementById(
        'deleteModal'
    ).style.display = 'none';
}


// =========================
// DELETE PDF
// =========================

async function confirmDeletePDF() {

    if (!deleteTargetPDF) return;


    await fetch(
        `/delete-pdf/${encodeURIComponent(deleteTargetPDF)}`,
        {
            method: 'DELETE'
        }
    );


    closeDeleteModal();


    if (selectedPDF === deleteTargetPDF) {

        selectedPDF = null;


        document.getElementById(
            'currentPDFText'
        ).innerText =
            'AI powered legal assistant';


        document.getElementById(
            'chatArea'
        ).innerHTML = `

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