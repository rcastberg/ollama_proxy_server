// Function to dynamically load and inject the sidebar HTML
window.onload = function() {
  console.log("Loading sidebar...");

  // Create a placeholder for the sidebar HTML
  const sidebarHTML = `
    <div id="sidebar">
      <ul>
        <!-- These links will be added conditionally -->
        <li id="user-admin-link" style="display: none;"><a href="/local/user_admin.html">User Admin</a></li>
        <li id="server-admin-link" style="display: none;"><a href="/local/server_admin.html">Server Admin</a></li>
        <li><a href="/local/user_info.html">User info</a></li>
        <li><a href="/local/show_usage.html">Statistics</a></li>
        <li><a href="/local/view_stats.html">User Statistics</a></li>
        <li><a href="/local/model_speed.html">Model Speed</a></li>
        <li><a href="/local/model_stats.html">Model statistics</a></li>
        <li><a href="/local/test_llm.html">Test LLM UI</a></li>
      </ul>
      <!-- Logout link at the bottom -->
      <div id="logout-link" style="position: absolute; bottom: 20px; width: 100%; text-align: center;">
        <a href="#" id="logout-text" style="color: white; text-decoration: none;">Logout</a>
      </div>
    </div>
    <style>
      #sidebar {
        width: 250px;
        background-color: #333;
        color: white;
        padding-top: 20px;
        height: 100%;
        position: fixed;
        top: 0;
        left: 0;
        flex-shrink: 0;
      }
      #sidebar ul {
        list-style-type: none;
        padding: 0;
      }
      #sidebar ul li {
        padding: 10px;
        text-align: center;
      }
      #sidebar ul li a {
        color: white;
        text-decoration: none;
        font-size: 18px;
      }
      #sidebar ul li a:hover {
        background-color: #575757;
        display: block;
        padding: 10px;
      }
      /* Wrapper to hold both sidebar and content */
      #wrapper {
        display: flex;
        min-height: 100vh;
      }
      /* Content section */
      #content {
        margin-left: 250px;
        padding: 20px;
        flex-grow: 1;
      }
      #logout-link {
        padding-top: 20px; /* Space from the bottom */
        padding-bottom: 20px; /* Space from the bottom */
        margin-top: auto; /* Push the logout to the bottom */
      }
    </style>
  `;

  // Insert the sidebar HTML into the document
  document.body.insertAdjacentHTML('afterbegin', sidebarHTML);

  // Make an AJAX call to /local/user_info to fetch the user's role
  fetch('/local/user_info')
    .then(response => response.json())
    .then(data => {
      // Check if the user is an admin
      if (data.role === 'admin') {
        // Show the admin links by removing the 'hidden' class
        document.getElementById('user-admin-link').style.display = 'block';
        document.getElementById('server-admin-link').style.display = 'block';
      } else {
        // Hide the admin links if the user is not an admin
        document.getElementById('user-admin-link').style.display = 'none';
        document.getElementById('server-admin-link').style.display = 'none';
      }
    })
    .catch(error => {
      console.error('Error fetching user info:', error);
    });
    // Logout functionality (text link)
  document.getElementById('logout-text').addEventListener('click', function(event) {
    event.preventDefault(); // Prevent the default action (link navigation)

    // Remove the auth_token cookie
    document.cookie = "auth_token=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;";

    // Refresh the page
    location.reload();
  });
};
