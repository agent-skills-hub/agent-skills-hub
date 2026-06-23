import XCTest

final class <#FeatureName#>UITests: XCTestCase {
    override func setUpWithError() throws {
        continueAfterFailure = false
    }

    func test_whenLaunchesFeature_thenCriticalFlowIsReachable() {
        let app = XCUIApplication()
        app.launchArguments.append("TEST-ONLY-UI")
        app.launch()

        XCTAssertTrue(app.<#criticalElement#>.exists)
    }

    func test_whenFormIsIncomplete_thenSaveButtonIsDisabled() {
        let app = XCUIApplication()
        app.launchArguments.append("TEST-ONLY-UI")
        app.launch()

        XCTAssertFalse(app.buttons["<#saveButtonIdentifier#>"].isEnabled)
    }

    func test_whenNoItemsAreSelected_thenBatchActionIsDisabled() {
        let app = XCUIApplication()
        app.launchArguments.append("TEST-ONLY-UI")
        app.launch()

        XCTAssertFalse(app.buttons["<#batchActionIdentifier#>"].isEnabled)
    }

    func test_whenPushedPageAppears_thenBackNavigationAffordanceExists() {
        let app = XCUIApplication()
        app.launchArguments.append("TEST-ONLY-UI")
        app.launch()

        app.buttons["<#openPushedPageIdentifier#>"].tap()

        XCTAssertTrue(app.navigationBars.buttons.element(boundBy: 0).exists)
    }
}
