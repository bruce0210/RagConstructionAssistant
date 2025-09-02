// =========================
// File: RAGca.cs
// Target: AutoCAD 2020 (64-bit), .NET Framework 4.7.2+ (recommend 4.8)
// Command: CGF
// Behavior: Opens a dockable Palette with an embedded browser that navigates to https://www.myarchibot.com
// Browser engine: WebView2 (modern Chromium). Falls back to WinForms WebBrowser if WebView2 cannot initialize.
// =========================

using System;
using System.Drawing;
using System.Threading.Tasks;
using System.Windows.Forms;
using Autodesk.AutoCAD.Runtime;
using Autodesk.AutoCAD.Windows;
using Autodesk.AutoCAD.ApplicationServices.Core;
// NuGet: Microsoft.Web.WebView2 (will add Microsoft.Web.WebView2.WinForms)
using Microsoft.Web.WebView2.WinForms;

namespace MyArchiBot.CGF
{
    // Simple palette control hosting a modern web view
    public class CgfPalette : UserControl
    {
        private const string TargetUrl = "https://www.myarchibot.com"; // change to https if available
        private Control _view;

        public CgfPalette()
        {
            this.Dock = DockStyle.Fill;
            this.Padding = new Padding(0);
            this.BackColor = SystemColors.Window;

            // Try WebView2 first (Chromium)
            try
            {
                var webView2 = new WebView2
                {
                    Dock = DockStyle.Fill
                };
                this.Controls.Add(webView2);
                _view = webView2;
                _ = InitWebView2Async(webView2);
            }
            catch
            {
                // Fallback: legacy IE-based control (works without extra runtime, but may break on modern sites)
                var wb = new WebBrowser
                {
                    Dock = DockStyle.Fill,
                    ScriptErrorsSuppressed = true
                };
                this.Controls.Add(wb);
                _view = wb;
                wb.Navigate(TargetUrl);
            }
        }

        private async Task InitWebView2Async(WebView2 wv2)
        {
            try
            {
                await wv2.EnsureCoreWebView2Async();
                wv2.Source = new Uri(TargetUrl);
            }
            catch
            {
                // If WebView2 runtime is missing or blocked, switch to the WinForms WebBrowser silently
                this.Controls.Remove(wv2);
                var wb = new WebBrowser
                {
                    Dock = DockStyle.Fill,
                    ScriptErrorsSuppressed = true
                };
                this.Controls.Add(wb);
                _view = wb;
                wb.Navigate(TargetUrl);
            }
        }
    }

    // AutoCAD entry point + command
    public class CgfPlugin : IExtensionApplication
    {
        private static PaletteSet _palette;

        public void Initialize()
        {
            // No-op; palette created on demand by the command.
        }

        public void Terminate()
        {
            try { _palette?.Dispose(); } catch { /* ignore */ }
        }

        [CommandMethod("CGF", CommandFlags.Modal)]
        public void OpenCgf()
        {
            if (_palette == null)
            {
                _palette = new PaletteSet("MyArchiBot – Code & Guide Finder")
                {
                    MinimumSize = new Size(420, 520),
                    Size = new Size(520, 680),
                    Style = PaletteSetStyles.ShowCloseButton
                          | PaletteSetStyles.ShowAutoHideButton
                          | PaletteSetStyles.ShowPropertiesMenu
                };
                _palette.Add("myarchibot.com", new CgfPalette());
            }

            _palette.Visible = true;
            _palette.KeepFocus = true;
        }
    }
}

/* =========================
BUILD NOTES (Visual Studio)
=========================
1) Create a Class Library (.NET Framework) project.
   - Target framework: .NET Framework 4.7.2 or 4.8
   - Platform target: x64 (uncheck "Prefer 32-bit").

2) Add references to AutoCAD 2020 managed assemblies (copy local = False):
   - acdbmgd.dll
   - acmgd.dll
   - AcCoreMgd.dll
   - AcWindows.dll
   These are in: C:\\Program Files\\Autodesk\\AutoCAD 2020\\ (or a nearby folder, e.g., \\Framework\\)

3) Add NuGet package: Microsoft.Web.WebView2 (stable). This adds Microsoft.Web.WebView2.WinForms.
   Ensure the WebView2Loader.dll (x64) is copied to the output folder (NuGet usually handles this).
   If the target PC doesn’t have the WebView2 Evergreen Runtime, install it from Microsoft.

4) Build Release x64. The output should include your DLL and WebView2 loader files.

5) Quick test inside AutoCAD 2020:
   - Command: NETLOAD → pick your built DLL → then type CGF → palette opens myarchibot.com

*/

// =========================
// (Optional) App Package for auto-load (recommended)
// Create the folder structure in %APPDATA%\Autodesk\ApplicationPlugins\MyArchiBot.CGF.bundle\
//   ├─ PackageContents.xml
//   └─ Contents\Windows\ MyArchiBot.CGF.dll + WebView2 loader DLLs
// =========================

/*
<?xml version="1.0" encoding="utf-8"?>
<ApplicationPackage SchemaVersion="1.0" 
    AutodeskProduct="AutoCAD" 
    Name="MyArchiBot.CGF" 
    Description="Open myarchibot.com inside a dockable palette (command: CGF)." 
    AppVersion="1.0" 
    ProductCode="{2B9A8E0E-2A4D-4D78-9E9A-1234567890AB}" 
    UpgradeCode="{7F6B3B36-7C8D-4E4F-AB45-0987654321CD}">
  <CompanyDetails Name="MyArchiBot" Url="https://www.myarchibot.com" />
  <Components>
    <RuntimeRequirements OS="Win64" Platform="AutoCAD" SeriesMin="R23.1" SeriesMax="R23.1"/>
    <ComponentEntry AppName="MyArchiBot.CGF" 
                    ModuleName="./Contents/Windows/MyArchiBot.CGF.dll" 
                    AppDescription="CGF palette" 
                    LoadOnAutoCADStartup="True">
      <Commands>
        <Command Local="CGF" Global="CGF"/>
      </Commands>
    </ComponentEntry>
  </Components>
</ApplicationPackage>
*/

// =========================
// (Optional) Pure IE-based fallback (no WebView2)
// Replace CgfPalette with the following minimal control if you cannot deploy WebView2.
// NOTE: Many modern sites may not render correctly under IE mode.
// =========================
/*
public class CgfPalette : UserControl
{
    private const string TargetUrl = "https://www.myarchibot.com";
    public CgfPalette()
    {
        var wb = new WebBrowser { Dock = DockStyle.Fill, ScriptErrorsSuppressed = true };
        Controls.Add(wb);
        wb.Navigate(TargetUrl);
    }
}
*/
